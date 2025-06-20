import torch
import torch.nn.functional as F
import numpy as np
import scipy.signal

def isolate_source(model, mixture, segment_size=16384, overlap=0.75, batch_size=1):
    """Process audio in segments with improved clarity."""
    device = next(model.parameters()).device
    model.eval()
    
    hop_size = int(segment_size * (1 - overlap))
    
    mixture_filtered = mixture
    
    # Pad mixture
    pad_left = segment_size - hop_size
    pad_right = segment_size - (mixture_filtered.shape[1] + pad_left) % hop_size
    mixture_padded = F.pad(mixture_filtered, (pad_left, pad_right), mode='reflect')
    
    # Create segments
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
            isolated_segments.append(output.cpu())
    
    isolated_segments = torch.cat(isolated_segments, dim=0)
    
    # Window and overlap-add
    window = torch.hann_window(segment_size, dtype=torch.float32)
    window = window.pow(0.75)
    
    result = torch.zeros(2, mixture_padded.shape[1], dtype=torch.float64)
    normalization = torch.zeros(1, mixture_padded.shape[1], dtype=torch.float64)
    
    for i in range(len(isolated_segments)):
        start_idx = i * hop_size
        end_idx = start_idx + segment_size
        result[:, start_idx:end_idx] += (isolated_segments[i] * window).to(torch.float64)
        normalization[0, start_idx:end_idx] += window.to(torch.float64)
    
    eps = 1e-10
    mask = normalization > eps
    result[:, mask[0]] /= normalization[0, mask[0]]
    
    result = result[:, pad_left:-pad_right if pad_right > 0 else None]
    result = result.to(torch.float32)
    
    return result