import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se_layer = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se_layer(x)

class CNNArchitecture(nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 64)):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            self.conv_block(1, 32),
            self.conv_block1(32, 64),
            self.conv_block1(64, 64),
            self.conv_block1(64, 128),
        )

        self.final_layer = nn.AdaptiveAvgPool3d(1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
    
    def conv_block1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.final_layer(x) 
        x = x.flatten(dim=1)
        return x 