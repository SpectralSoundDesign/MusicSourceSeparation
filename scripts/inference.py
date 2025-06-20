import torch
import torchaudio
from models.wave_unet import WaveUNet
from processing.source_isolation import isolate_source


def load_audio(audio_path, sample_rate=44100):
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 2:
        waveform = waveform[:2]
    elif waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(
            sr, sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475,
            resampling_method="sinc_interpolation"
        )
        waveform = resampler(waveform)
    
    waveform = waveform - torch.mean(waveform, dim=1, keepdim=True)
    
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0.95:
        waveform = 0.95 * waveform / max_val
    
    return waveform


def main():
    model = WaveUNet(
        num_layers=10,
        num_initial_filters=16,
        kernel_size=5,
        input_channels=2,
        output_channels=2,
        merge_filter_size=5
    )
    
    checkpoint_path = "checkpoints/best_model.pt"
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    input_path = "mix.wav"
    output_path = "isolated_bass.wav"
    
    print(f"Loading audio file: {input_path}")
    mixture = load_audio(input_path)
    
    print("Isolating bass...")
    isolated_bass = isolate_source(
        model, 
        mixture, 
        segment_size=16384,
        overlap=0.75
    )
    
    isolated_bass = isolated_bass.to(dtype=torch.float32)
    
    print(f"Saving result to: {output_path}")
    torchaudio.save(output_path, isolated_bass, 44100)
    print("Done!")


if __name__ == "__main__":
    main()