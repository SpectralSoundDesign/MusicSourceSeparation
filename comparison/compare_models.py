"""
Compare performance of Wave-U-Net and other models for bass isolation.
"""

import os
import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import musdb
from demucs.pretrained import get_model
from demucs.separate import load_track, save_audio, apply_model

# Import from your existing modules
from models.wave_unet import WaveUNet
from processing.source_isolation import isolate_source
from evaluation.metrics import compute_all_metrics


def load_waveunet_model(checkpoint_path, device):
    """Load and prepare Wave-U-Net model."""
    print(f"Loading Wave-U-Net model from {checkpoint_path}")
    model = WaveUNet(
        num_layers=10,
        num_initial_filters=16,
        kernel_size=5,
        input_channels=2,
        output_channels=2,
        merge_filter_size=5
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_demucs_model(device):
    """Load and prepare Demucs model."""
    print("Loading Demucs model")
    model = get_model("htdemucs")
    model.to(device)
    model.eval()
    return model


def process_with_waveunet(model, mixture, device, segment_size=16384, overlap=0.75):
    """Process a mixture with Wave-U-Net model."""
    # Ensure correct input format
    if len(mixture.shape) == 3:  # [batch, channels, samples]
        mixture = mixture[0]  # Remove batch dimension
    
    mixture = mixture.to(device)
    
    with torch.no_grad():
        isolated_bass = isolate_source(
            model,
            mixture,
            segment_size=segment_size,
            overlap=overlap
        )
    
    # Move tensor to CPU before any potential conversion to NumPy
    isolated_bass = isolated_bass.cpu()
    
    return isolated_bass


def process_with_demucs(model, mixture, device):
    """Process a mixture with Demucs model specifically for bass extraction."""
    # Add batch dimension if missing
    mixture = mixture.unsqueeze(0) if len(mixture.shape) == 2 else mixture
    
    mixture = mixture.to(device)
    
    with torch.no_grad():
        # Apply the model to separate sources
        sources = apply_model(model, mixture)
    
    # Assuming `sources.shape == [1, 4, 2, N]`
    bass_index = model.sources.index('bass')
    isolated_bass = sources[0, bass_index]  # [2, N]

    
    # Remove batch dimension
    isolated_bass = isolated_bass.squeeze(0)

    # Move tensor to CPU before any potential conversion to NumPy
    isolated_bass = isolated_bass.cpu()
    
    return isolated_bass


def evaluate_on_musdb18(waveunet_checkpoint, output_dir, num_tracks=None):
    """
    Compare Wave-U-Net and Demucs models on MUSDB18 test set.
    
    Args:
        waveunet_checkpoint: Path to Wave-U-Net model checkpoint
        output_dir: Directory to save results
        num_tracks: Number of tracks to process (None for all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio_samples")
    os.makedirs(audio_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    waveunet = load_waveunet_model(waveunet_checkpoint, device)
    demucs = load_demucs_model(device)
    
    # Setup MUSDB
    mus = musdb.DB(root="./data/musdb18", subsets="test")
    
    # Limit number of tracks if specified
    if num_tracks is not None:
        tracks = list(mus.tracks)[:num_tracks]
    else:
        tracks = list(mus.tracks)
    
    # Prepare results storage
    results = {
        "waveunet": {
            "tracks": [],
            "average": {"SDR": 0, "SI-SDR": 0, "SNR": 0}
        },
        "demucs": {
            "tracks": [],
            "average": {"SDR": 0, "SI-SDR": 0, "SNR": 0}
        }
    }
    
    # Process each track
    for i, track in enumerate(tqdm(tracks, desc="Processing tracks")):
        track_name = track.name
        print(f"\nProcessing track {i+1}/{len(tracks)}: {track_name}")
        
        # Get mixture and ground truth
        mixture = torch.tensor(track.audio.T, dtype=torch.float32)  # Convert to channels-first
        target_bass = torch.tensor(track.targets['bass'].audio.T, dtype=torch.float32)
        
        # Print debug info
        print(f"Mixture shape: {mixture.shape}, Target shape: {target_bass.shape}")
        
        # Normalize
        mixture = mixture / (torch.max(torch.abs(mixture)) + 1e-8)
        target_bass = target_bass / (torch.max(torch.abs(target_bass)) + 1e-8)
        
        # Move to CPU for saving
        mixture_cpu = mixture.cpu()
        target_bass_cpu = target_bass.cpu()
        
        # Process with both models
        waveunet_bass = process_with_waveunet(waveunet, mixture, device)
        demucs_bass = process_with_demucs(demucs, mixture, device)
        
        # Debug shapes
        print(f"WaveUNet output shape: {waveunet_bass.shape}")
        print(f"Demucs output shape: {demucs_bass.shape}")
        
        # Ensure Demucs output matches WaveUNet and target shape
        if demucs_bass.shape[0] != target_bass_cpu.shape[0]:
            # If the channel dimension doesn't match (e.g., 4 channels vs 2), take only needed channels
            if demucs_bass.shape[0] > target_bass_cpu.shape[0]:
                demucs_bass = demucs_bass[:target_bass_cpu.shape[0]]
            else:
                # If Demucs output has fewer channels, duplicate the channels
                demucs_bass = demucs_bass.repeat(target_bass_cpu.shape[0] // demucs_bass.shape[0], 1)

        min_length = min(waveunet_bass.shape[1], demucs_bass.shape[1], target_bass_cpu.shape[1])
        waveunet_bass = waveunet_bass[:, :min_length]
        demucs_bass = demucs_bass[:, :min_length]
        target_bass_cpu = target_bass_cpu[:, :min_length]

        
        # Final shape check
        print(f"After adjustment - WaveUNet: {waveunet_bass.shape}, Demucs: {demucs_bass.shape}, Target: {target_bass_cpu.shape}")

        

        # Save audio samples (first 30 seconds)
        sample_length = min(44100 * 30, min_length)
        
        # Save mixture
        torchaudio.save(
            os.path.join(audio_dir, f"{track_name}_mixture.wav"),
            mixture_cpu[:, :sample_length],
            44100
        )
        
        # Save target
        torchaudio.save(
            os.path.join(audio_dir, f"{track_name}_target.wav"),
            target_bass_cpu[:, :sample_length],
            44100
        )
        
        # Save Wave-U-Net output
        torchaudio.save(
            os.path.join(audio_dir, f"{track_name}_waveunet.wav"),
            waveunet_bass[:, :sample_length],
            44100
        )
        # Save Demucs output
        torchaudio.save(
            os.path.join(audio_dir, f"{track_name}_demucs.wav"),
            demucs_bass[:, :sample_length],
            44100
        )
        
        # Compute metrics - ensure dimensions match
        try:
            waveunet_metrics = compute_all_metrics(waveunet_bass, target_bass_cpu)
            demucs_metrics = compute_all_metrics(demucs_bass, target_bass_cpu)
            
            # Store track results
            track_result_waveunet = {
                "track_name": track_name,
                "metrics": waveunet_metrics
            }
            
            track_result_demucs = {
                "track_name": track_name,
                "metrics": demucs_metrics
            }
            
            results["waveunet"]["tracks"].append(track_result_waveunet)
            results["demucs"]["tracks"].append(track_result_demucs)
            
            # Print track metrics
            print(f"Track: {track_name}")
            print(f"  Wave-U-Net: SDR={waveunet_metrics['SDR']:.2f} dB, SI-SDR={waveunet_metrics['SI-SDR']:.2f} dB")
            print(f"  Demucs: SDR={demucs_metrics['SDR']:.2f} dB, SI-SDR={demucs_metrics['SI-SDR']:.2f} dB")
        
        except Exception as e:
            print(f"Error computing metrics for track {track_name}: {e}")
            # Add placeholder with None values
            track_result = {
                "track_name": track_name,
                "metrics": {"SDR": None, "SI-SDR": None, "SNR": None}
            }
            results["waveunet"]["tracks"].append(track_result)
            results["demucs"]["tracks"].append(track_result)
    
    # Calculate average metrics
    for model in ["waveunet", "demucs"]:
        valid_metrics = {
            "SDR": [],
            "SI-SDR": [],
            "SNR": []
        }
        
        for track_result in results[model]["tracks"]:
            for metric_name, value in track_result["metrics"].items():
                if value is not None and not np.isinf(value) and not np.isnan(value):
                    valid_metrics[metric_name].append(value)
        
        # Compute averages
        for metric_name, values in valid_metrics.items():
            if values:
                results[model]["average"][metric_name] = sum(values) / len(values)
    
    # Save detailed results
    with open(os.path.join(output_dir, "comparison_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nComparison Results Summary:")
    print("===========================")
    for model in ["waveunet", "demucs"]:
        print(f"\n{model.upper()}:")
        for metric, value in results[model]["average"].items():
            print(f"  {metric}: {value:.2f} dB")
    
    print(f"\nDetailed results saved to: {os.path.join(output_dir, 'comparison_results.json')}")
    print(f"Audio samples saved to: {audio_dir}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Wave-U-Net with other models")
    parser.add_argument('--waveunet_checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to Wave-U-Net checkpoint')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Directory to save results')
    parser.add_argument('--num_tracks', type=int, default=None,
                        help='Number of tracks to evaluate (None for all)')
    
    args = parser.parse_args()
    
    evaluate_on_musdb18(
        args.waveunet_checkpoint,
        args.output_dir,
        args.num_tracks
    )


if __name__ == "__main__":
    main()