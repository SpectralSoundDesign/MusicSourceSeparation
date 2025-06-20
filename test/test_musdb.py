import pytest
import torch
import os
from datasets.musdb_dataset import MUSDB18Dataset


# Skip if MUSDB not available
pytestmark = pytest.mark.skipif(
    not os.path.exists('./data/musdb18'),
    reason="MUSDB dataset not found in ./data/musdb18"
)


def test_musdb_loading():
    """Test loading MUSDB dataset."""
    dataset = MUSDB18Dataset('./data/musdb18', 'test', segment_length=8192)
    
    # Check dataset exists
    assert len(dataset) > 0
    
    # Load one sample
    mixture, bass = dataset[0]
    
    # Check shapes and values
    assert mixture.shape == (2, 8192)
    assert bass.shape == (2, 8192)
    assert -1.1 <= mixture.min() <= mixture.max() <= 1.1
    assert -1.1 <= bass.min() <= bass.max() <= 1.1