import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
from models.wave_unet import WaveUNet
from datasets.musdb_dataset import MUSDB18Dataset


def create_data_loaders(root_dir, batch_size=2, segment_length=8192, 
                       valid_split=0.1, num_workers=2):
    full_train_dataset = MUSDB18Dataset(root_dir, 'train', segment_length)
    
    total_size = len(full_train_dataset)
    valid_size = int(total_size * valid_split)
    train_size = total_size - valid_size
    
    train_dataset, valid_dataset = random_split(
        full_train_dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = MUSDB18Dataset(root_dir, 'test', segment_length)
    
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
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    best_valid_loss = float('inf')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_valid_loss = checkpoint.get('loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming training from epoch {start_epoch}...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (mixture, vocals) in enumerate(pbar):
            mixture, vocals = mixture.to(device), vocals.to(device)
            
            optimizer.zero_grad()
            
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
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_valid_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
        
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))


def main():
    model = WaveUNet(
        num_layers=10,
        num_initial_filters=16,
        kernel_size=5,
        input_channels=2,
        output_channels=2,
        merge_filter_size=5
    )
    
    train_loader, valid_loader, test_loader = create_data_loaders(
        root_dir='./data/musdb18',
        batch_size=2,
        segment_length=8192,
        valid_split=0.1,
        num_workers=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = None

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