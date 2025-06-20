import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import musdb
import random
from pathlib import Path


class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir, split='train', segment_length=8192, 
                 sample_rate=44100):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        self.mus = musdb.DB(root=str(root_dir), subsets=split)
        print(f"Found {len(self.mus)} tracks in {split} set")

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, idx):
        track = self.mus[idx]
        
        mixture = torch.tensor(track.audio.T, dtype=torch.float32)
        vocals = torch.tensor(track.targets['bass'].audio.T, dtype=torch.float32)
        
        # Random segment selection
        if mixture.shape[1] > self.segment_length:
            start = random.randint(0, mixture.shape[1] - self.segment_length)
            mixture = mixture[:, start:start + self.segment_length]
            vocals = vocals[:, start:start + self.segment_length]
        else:
            pad_length = self.segment_length - mixture.shape[1]
            mixture = F.pad(mixture, (0, pad_length))
            vocals = F.pad(vocals, (0, pad_length))
        
        # Normalize
        mixture = mixture / (torch.max(torch.abs(mixture)) + 1e-6)
        vocals = vocals / (torch.max(torch.abs(vocals)) + 1e-6)
        
        return mixture, vocals