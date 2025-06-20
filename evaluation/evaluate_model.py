import torch
import os
import json
import numpy as np
from tqdm import tqdm
from models.wave_unet import WaveUNet
from datasets.musdb_dataset import MUSDB18Dataset
from processing.source_isolation import isolate_source
from evaluation.metrics import compute_all_metrics


def evaluate_on_musdb(model_path, output_dir='evaluation_results'):
    """
    Evaluate model on MUSDB test set.
    
    Args:
        model_path: Path to trained model checkpoint
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = WaveUNet(
        num_layers=10,
        num_initial_filters=16,
        kernel_size=5,
        input_channels=2,
        output_channels=2,
        merge_filter_size=5
    )
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    print("Loading MUSDB test set...")
    test_dataset = MUSDB18Dataset(
        root_dir='./data/musdb18',
        split='test',
        segment_length=44100 * 10  # 10 second segments for evaluation
    )
    
    # Evaluate on each track
    results = []
    all_metrics = {
        'SDR': [],
        'SI-SDR': [],
        'SNR': []
    }
    
    print(f"Evaluating on {len(test_dataset)} test tracks...")
    for i in tqdm(range(len(test_dataset))):
        mixture, target_bass = test_dataset[i]
        
        # Process with model
        with torch.no_grad():
            isolated_bass = isolate_source(
                model,
                mixture,
                segment_size=16384,
                overlap=0.75,
                batch_size=1
            )
        
        # Debug output
        if i == 0:  # Check first track
            print(f"\nDebug info for first track:")
            print(f"Mixture shape: {mixture.shape}")
            print(f"Target bass shape: {target_bass.shape}")
            print(f"Isolated bass shape: {isolated_bass.shape}")
            print(f"Target bass energy: {torch.sum(target_bass ** 2).item()}")
            print(f"Isolated bass energy: {torch.sum(isolated_bass ** 2).item()}")
            print(f"Contains NaN: isolated={torch.isnan(isolated_bass).any()}, target={torch.isnan(target_bass).any()}")
        
        # Compute metrics
        metrics = compute_all_metrics(isolated_bass, target_bass)
        
        # Filter out inf and nan values for averaging
        valid_metrics = {}
        for metric_name, value in metrics.items():
            if not np.isinf(value) and not np.isnan(value):
                valid_metrics[metric_name] = value
                all_metrics[metric_name].append(value)
            else:
                valid_metrics[metric_name] = None
        
        # Store results
        results.append({
            'track_id': i,
            'metrics': valid_metrics
        })
    
    # Compute average metrics (excluding inf and nan)
    avg_metrics = {}
    for metric, values in all_metrics.items():
        if values:  # If we have valid values
            avg_metrics[metric] = sum(values) / len(values)
        else:
            avg_metrics[metric] = float('nan')
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'individual_tracks': results,
            'average_metrics': avg_metrics
        }, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print("------------------")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.2f} dB")
    
    print(f"\nDetailed results saved to: {results_path}")
    
    return avg_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on MUSDB test set")
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    evaluate_on_musdb(args.model_path, args.output_dir)