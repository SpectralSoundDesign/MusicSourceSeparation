import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import musdb
import random

class WaveUNet(nn.Module):
    def __init__(self, num_layers=12, num_initial_filters=16, kernel_size=5, 
                 input_channels=2, output_channels=2, merge_filter_size=5):
        super(WaveUNet, self).__init__()
        
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.merge_filter_size = merge_filter_size
        self.num_initial_filters = num_initial_filters

        # Store all channel sizes for proper skip connections
        self.channel_sizes = []
        
        # Downsampling path
        self.downsample_blocks = nn.ModuleList()
        current_channels = input_channels
        
        for i in range(num_layers):
            out_channels = min(num_initial_filters * (2**i), 512)
            self.downsample_blocks.append(
                nn.Sequential(
                    nn.Conv1d(current_channels, out_channels, kernel_size, 
                             padding=(kernel_size-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(out_channels)
                )
            )
            self.channel_sizes.append(out_channels)
            current_channels = out_channels

        # Bottleneck
        bottleneck_channels = min(current_channels*2, 512)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(current_channels, bottleneck_channels, kernel_size, 
                     padding=(kernel_size-1)//2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(bottleneck_channels)
        )

        # Upsampling path
        self.upsample_blocks = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        
        current_channels = bottleneck_channels
        
        for i in range(num_layers):
            # Calculate input and output channels for each level
            if i == 0:
                merge_input_channels = current_channels
            else:
                merge_input_channels = current_channels + self.channel_sizes[-i]
                
            out_channels = self.channel_sizes[-i-1] if i < num_layers-1 else num_initial_filters
            
            # Upsampling block
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(current_channels, out_channels, 
                                     kernel_size, padding=(kernel_size-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(out_channels)
                )
            )
            
            # Merge block
            if i < num_layers - 1:
                self.merge_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(out_channels + self.channel_sizes[-i-1], 
                                out_channels, merge_filter_size, 
                                padding=(merge_filter_size-1)//2),
                        nn.LeakyReLU(0.2),
                        nn.BatchNorm1d(out_channels)
                    )
                )
            
            current_channels = out_channels

        # Final output layer
        self.output_layer = nn.Conv1d(current_channels, output_channels, 1)

    def forward(self, x):
        # Store intermediate outputs for skip connections
        intermediates = []
        
        # Downsampling
        current = x
        for block in self.downsample_blocks:
            current = block(current)
            intermediates.append(current)
            current = F.avg_pool1d(current, 2)

        # Bottleneck
        current = self.bottleneck(current)

        # Upsampling
        for i, (up_block, merge_block) in enumerate(zip(self.upsample_blocks[:-1], 
                                                      self.merge_blocks)):
            # Upsample
            current = F.interpolate(current, scale_factor=2, mode='linear', 
                                  align_corners=False)
            current = up_block(current)
            
            # Get corresponding skip connection
            skip = intermediates[-i-1]
            
            # Ensure matching sizes for concatenation
            if current.size(-1) != skip.size(-1):
                current = F.interpolate(current, size=skip.size(-1), 
                                     mode='linear', align_corners=False)
            
            # Concatenate and merge
            current = torch.cat([current, skip], dim=1)
            current = merge_block(current)

        # Final upsampling
        current = F.interpolate(current, scale_factor=2, mode='linear', 
                              align_corners=False)
        current = self.upsample_blocks[-1](current)
        
        return self.output_layer(current)

class MUSDB18Dataset(Dataset):
    def __init__(self, root_dir, split='train', segment_length=8192, 
                 sample_rate=44100):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        
        # Initialize MUSDB
        self.mus = musdb.DB(root=str(root_dir), subsets=split)
        print(f"Found {len(self.mus)} tracks in {split} set")

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, idx):
        track = self.mus[idx]
        
        # Load and convert to torch tensors
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

def create_data_loaders(root_dir, batch_size=2, segment_length=8192, 
                       valid_split=0.1, num_workers=2):
    # Create the full training dataset
    full_train_dataset = MUSDB18Dataset(root_dir, 'train', segment_length)
    
    # Calculate split sizes
    total_size = len(full_train_dataset)
    valid_size = int(total_size * valid_split)
    train_size = total_size - valid_size
    
    # Split into train and validation sets
    train_dataset, valid_dataset = random_split(
        full_train_dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create test dataset
    test_dataset = MUSDB18Dataset(root_dir, 'test', segment_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader

def train_wave_unet(model, train_loader, valid_loader, num_epochs=100, 
                    learning_rate=0.001, device='cuda', checkpoint_dir='checkpoints', 
                    checkpoint_path=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                         patience=5, factor=0.5)
    criterion = nn.L1Loss()
    
    # Enable automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Load checkpoint if provided
    start_epoch = 0
    best_valid_loss = float('inf')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_valid_loss = checkpoint.get('loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1  # Continue from next epoch
        print(f"Resuming training from epoch {start_epoch}...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (mixture, vocals) in enumerate(pbar):
            mixture, vocals = mixture.to(device), vocals.to(device)
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():
                output = model(mixture)
                loss = criterion(output, vocals)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for mixture, vocals in valid_loader:
                mixture, vocals = mixture.to(device), vocals.to(device)
                with torch.cuda.amp.autocast():
                    output = model(mixture)
                    valid_loss += criterion(output, vocals).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        scheduler.step(avg_valid_loss)
        
        print(f'Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_valid_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

def main():
    # Initialize model
    model = WaveUNet(
        num_layers=10,
        num_initial_filters=16,
        kernel_size=5,
        input_channels=2,
        output_channels=2,
        merge_filter_size=5
    )
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        root_dir='./data/musdb18',
        batch_size=2,
        segment_length=8192,
        valid_split=0.1,
        num_workers=2
    )
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Path to checkpoint (set to None to train from scratch)
    checkpoint_path = None  # Change this as needed

    # Train the model
    train_wave_unet(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=100,
        learning_rate=0.001,
        device=device,
        checkpoint_dir='checkpoints',
        checkpoint_path=checkpoint_path
    )

if __name__ == "__main__":
    main()

