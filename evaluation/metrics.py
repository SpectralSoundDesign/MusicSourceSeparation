import torch
import numpy as np


def compute_sdr(estimated, reference):
    """
    Compute Signal-to-Distortion Ratio.
    
    Args:
        estimated: Estimated source signal
        reference: Reference source signal
    
    Returns:
        SDR in dB
    """
    # Handle edge cases
    if torch.isnan(estimated).any() or torch.isnan(reference).any():
        return float('nan')
    
    reference_energy = torch.sum(reference ** 2)
    if reference_energy < 1e-10:
        return float('nan')
    
    error = estimated - reference
    error_energy = torch.sum(error ** 2)
    
    if error_energy < 1e-10:
        return float('inf')  # Perfect reconstruction
    
    sdr = 10 * torch.log10(reference_energy / error_energy)
    return sdr.item()


def compute_si_sdr(estimated, reference):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio.
    
    Args:
        estimated: Estimated source signal
        reference: Reference source signal
    
    Returns:
        SI-SDR in dB
    """
    # Handle edge cases
    if torch.isnan(estimated).any() or torch.isnan(reference).any():
        return float('nan')
    
    ref_energy = torch.sum(reference ** 2)
    if ref_energy < 1e-10:
        return float('nan')
    
    alpha = torch.sum(estimated * reference) / ref_energy
    reference_scaled = alpha * reference
    noise = estimated - reference_scaled
    
    noise_energy = torch.sum(noise ** 2)
    if noise_energy < 1e-10:
        return float('inf')  # Perfect reconstruction
    
    si_sdr = 10 * torch.log10(torch.sum(reference_scaled ** 2) / noise_energy)
    return si_sdr.item()


def compute_snr(estimated, reference):
    """
    Compute Signal-to-Noise Ratio.
    
    Args:
        estimated: Estimated source signal
        reference: Reference source signal
    
    Returns:
        SNR in dB
    """
    # Handle edge cases
    if torch.isnan(estimated).any() or torch.isnan(reference).any():
        return float('nan')
    
    ref_energy = torch.sum(reference ** 2)
    if ref_energy < 1e-10:
        return float('nan')
    
    noise = estimated - reference
    noise_energy = torch.sum(noise ** 2)
    
    if noise_energy < 1e-10:
        return float('inf')  # Perfect reconstruction
    
    snr = 10 * torch.log10(ref_energy / noise_energy)
    return snr.item()


def compute_all_metrics(estimated, reference):
    """
    Compute all audio quality metrics.
    
    Args:
        estimated: Estimated source signal
        reference: Reference source signal
    
    Returns:
        Dictionary of metrics
    """
    return {
        'SDR': compute_sdr(estimated, reference),
        'SI-SDR': compute_si_sdr(estimated, reference),
        'SNR': compute_snr(estimated, reference)
    }