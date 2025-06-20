import pytest
import torch
from models.wave_unet import WaveUNet


def test_model_forward():
    """Test that model forward pass works correctly."""
    model = WaveUNet(
        num_layers=3,
        num_initial_filters=8,
        kernel_size=3,
        input_channels=2,
        output_channels=2
    )
    
    # Test input
    batch_size = 2
    channels = 2
    length = 1024
    input_tensor = torch.randn(batch_size, channels, length)
    
    # Forward pass
    output = model(input_tensor)
    
    # Check output shape
    assert output.shape == (batch_size, 2, length)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    model = WaveUNet(num_layers=2, num_initial_filters=4)
    
    input_tensor = torch.randn(1, 2, 512, requires_grad=True)
    output = model(input_tensor)
    loss = output.mean()
    loss.backward()
    
    # Check gradients exist
    assert input_tensor.grad is not None
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None