"""
MRI Encoder Module.
Uses a 3D CNN to extract features from 3D MRI volumes (.nii).
"""
import torch
import torch.nn as nn

class MRIEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 128):
        super(MRIEncoder, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling to ensure a fixed size before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc = nn.Linear(64 * 4 * 4 * 4, out_features)
        self.feature_dim = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 3D MRI Tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            Embeddings of shape (batch_size, out_features)
        """
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        embeddings = self.fc(x)
        return embeddings
