import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveUNet(nn.Module):
    def __init__(self, num_layers=12, num_initial_filters=16, kernel_size=5, 
                 input_channels=2, output_channels=2, merge_filter_size=5):
        super(WaveUNet, self).__init__()
        
        self.num_layers = num_layers
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
            out_channels = self.channel_sizes[-i-1] if i < num_layers-1 else num_initial_filters
            
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(current_channels, out_channels, 
                                     kernel_size, padding=(kernel_size-1)//2),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(out_channels)
                )
            )
            
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

        self.output_layer = nn.Conv1d(current_channels, output_channels, 1)

    def forward(self, x):
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
            current = F.interpolate(current, scale_factor=2, mode='linear', 
                                  align_corners=False)
            current = up_block(current)
            
            skip = intermediates[-i-1]
            if current.size(-1) != skip.size(-1):
                current = F.interpolate(current, size=skip.size(-1), 
                                     mode='linear', align_corners=False)
            
            current = torch.cat([current, skip], dim=1)
            current = merge_block(current)

        # Final upsampling
        current = F.interpolate(current, scale_factor=2, mode='linear', 
                              align_corners=False)
        current = self.upsample_blocks[-1](current)
        
        return self.output_layer(current)