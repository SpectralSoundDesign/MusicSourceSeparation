import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
import numpy as np
import scipy.signal
import torch.fft
from main_improved import WaveUNet

def enhance_drums_torch(audio):
    """Enhance drums using PyTorch operations with optimized frequency bands."""
    n_fft = 4096
    hop_length = n_fft // 4
    window = torch.hann_window(n_fft).to(audio.device)
    
    # Reshape audio for STFT processing
    batch_size, channels, samples = audio.shape
    audio_reshape = audio.view(batch_size * channels, samples)
    
    # Compute STFT
    spec = torch.stft(
        audio_reshape,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    
    # Get dimensions
    num_freqs = spec.shape[1]
    
    # Create frequency masks
    freq_resolution = 44100 / n_fft
    freqs = torch.linspace(0, 44100/2, num_freqs).to(audio.device)
    
    # Define frequency bands for drum components
    sub_mask = (freqs <= 60).float()
    kick_mask = ((freqs > 60) & (freqs <= 150)).float()
    snare_mask = ((freqs > 150) & (freqs <= 500)).float()
    cymbal_mask = (freqs > 500).float()

    # Process magnitudes
    spec_mag = torch.abs(spec)
    spec_phase = torch.angle(spec)
    
    # Reshape masks for broadcasting
    sub_mask = sub_mask.view(1, -1, 1)      # [1, freq, 1]
    kick_mask = kick_mask.view(1, -1, 1)    # [1, freq, 1]
    snare_mask = snare_mask.view(1, -1, 1)  # [1, freq, 1]
    cymbal_mask = cymbal_mask.view(1, -1, 1)# [1, freq, 1]
    
    # Apply frequency-dependent processing with proper broadcasting
    enhancement = (sub_mask * 0.7 +
                  kick_mask * 1.3 +
                  snare_mask * 1.4 +
                  cymbal_mask * 1.1)
    
    spec_mag = spec_mag * enhancement.expand_as(spec_mag)
    
    # Enhance transients
    spec_diff = torch.diff(spec_mag, dim=-1, prepend=spec_mag[..., :1])
    transient_mask = (spec_diff > 0.1 * torch.mean(spec_diff)).float()
    spec_mag = spec_mag * (1.0 + 0.3 * transient_mask)
    
    # Reconstruct complex spectrum
    processed_spec = spec_mag * torch.exp(1j * spec_phase)
    
    # Inverse STFT
    enhanced = torch.istft(
        processed_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=samples
    )
    
    # Reshape back to original dimensions
    enhanced = enhanced.view(batch_size, channels, samples)
    
    return enhanced

def enhance_drums_final(audio):
    """Final processing stage with improved drum enhancement."""
    # Dynamic range compression
    threshold = 0.5
    ratio = 0.7
    makeup_gain = 1.2
    
    abs_audio = torch.abs(audio)
    gain_reduction = torch.where(
        abs_audio > threshold,
        (abs_audio - threshold) * (1 - ratio),
        torch.zeros_like(abs_audio)
    )
    
    audio_compressed = torch.sign(audio) * (abs_audio - gain_reduction)
    audio_compressed = audio_compressed * makeup_gain
    
    # Attack enhancement with proper dimensions
    audio_diff = torch.diff(audio_compressed, dim=-1, prepend=audio_compressed[..., :1])
    attack_enhancement = torch.sign(audio_diff) * torch.pow(torch.abs(audio_diff), 0.7)
    
    # Mix enhanced attack with compressed signal
    audio_final = audio_compressed + 0.2 * attack_enhancement
    
    # Final limiting and normalization
    audio_final = torch.tanh(audio_final)
    
    max_val = torch.max(torch.abs(audio_final))
    if max_val > 0.9:
        audio_final = 0.9 * audio_final / max_val
    
    return audio_final

def enhance_drums_final(audio):
    """Final processing stage with improved drum enhancement."""
    threshold = 0.5
    ratio = 0.7
    makeup_gain = 1.2
    
    abs_audio = torch.abs(audio)
    gain_reduction = torch.where(
        abs_audio > threshold,
        (abs_audio - threshold) * (1 - ratio),
        torch.zeros_like(abs_audio)
    )
    
    audio_compressed = torch.sign(audio) * (abs_audio - gain_reduction)
    audio_compressed = audio_compressed * makeup_gain
    
    audio_diff = torch.diff(audio_compressed, dim=1, prepend=audio_compressed[:, :1])
    attack_enhancement = torch.sign(audio_diff) * torch.pow(torch.abs(audio_diff), 0.7)
    
    audio_final = audio_compressed + 0.2 * attack_enhancement
    audio_final = torch.tanh(audio_final)
    
    max_val = torch.max(torch.abs(audio_final))
    if max_val > 0.9:
        audio_final = 0.9 * audio_final / max_val
    
    return audio_final

def isolate_source(model, mixture, segment_size=16384, overlap=0.75, batch_size=1):
    """Process audio in segments with enhanced drum clarity."""
    device = next(model.parameters()).device
    model.eval()
    
    hop_size = int(segment_size * (1 - overlap))
    margin = segment_size // 4
    
    # High-pass filter for rumble control
    cutoff_freq = 30
    nyquist = 44100 / 2
    filter_order = 4
    wn = cutoff_freq / nyquist
    b, a = scipy.signal.butter(filter_order, wn, btype='high')
    
    # Ensure contiguous memory layout throughout the conversion process
    mixture_np = mixture.cpu().numpy().copy()  # Ensure it's on CPU and contiguous
    mixture_filtered_np = scipy.signal.filtfilt(b, a, mixture_np, axis=1)
    mixture_filtered_np = np.ascontiguousarray(mixture_filtered_np)  # Ensure contiguous layout
    mixture_filtered = torch.from_numpy(mixture_filtered_np).to(mixture.dtype)
    
    pad_left = segment_size - hop_size
    pad_right = segment_size - (mixture_filtered.shape[1] + pad_left) % hop_size
    mixture_padded = F.pad(mixture_filtered, (pad_left, pad_right), mode='reflect')
    
    segments = []
    for i in range(0, mixture_padded.shape[1] - segment_size + 1, hop_size):
        segment = mixture_padded[:, i:i + segment_size]
        segments.append(segment)
    segments = torch.stack(segments)
    
    # Process segments
    isolated_segments = []
    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size].to(device)
            output = model(batch)
            output = enhance_drums_torch(output)
            isolated_segments.append(output.cpu())
    
    isolated_segments = torch.cat(isolated_segments, dim=0)
    
    # Window and overlap-add processing
    window = torch.hann_window(segment_size, dtype=torch.float32)
    window = window.pow(0.75)
    
    result = torch.zeros(2, mixture_padded.shape[1], dtype=torch.float64)
    normalization = torch.zeros(1, mixture_padded.shape[1], dtype=torch.float64)
    
    for i in range(len(isolated_segments)):
        start_idx = i * hop_size
        end_idx = start_idx + segment_size
        result[:, start_idx:end_idx] += (isolated_segments[i] * window).to(torch.float64)
        normalization[0, start_idx:end_idx] += window.to(torch.float64)
    
    # Normalize and finalize
    eps = 1e-10
    mask = normalization > eps
    result[:, mask[0]] /= normalization[0, mask[0]]
    
    result = result[:, pad_left:-pad_right if pad_right > 0 else None]
    result = result.to(torch.float32)
    
    result = enhance_drums_final(result)
    
    return result

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
    output_path = "isolated_drums.wav"  # Changed from vocals to drums
    
    print(f"Loading audio file: {input_path}")
    mixture = load_audio(input_path)
    
    print("Isolating drums...")
    isolated_drums = isolate_source(
        model, 
        mixture, 
        segment_size=16384,
        overlap=0.75
    )
    
    isolated_drums = isolated_drums.to(dtype=torch.float32)
    
    print(f"Saving result to: {output_path}")
    torchaudio.save(output_path, isolated_drums, 44100)
    print("Done!")

if __name__ == "__main__":
    main()