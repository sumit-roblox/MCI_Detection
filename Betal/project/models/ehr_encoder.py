"""
EHR Encoder Module.
Uses a small MLP to extract features from tabular patient data.
"""
import torch
import torch.nn as nn

class EHREncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int = 64):
        super(EHREncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, out_features),
            nn.ReLU()
        )
        self.feature_dim = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EHR tabular features Tensor of shape (batch_size, in_features)
        Returns:
            Embeddings of shape (batch_size, out_features)
        """
        embeddings = self.mlp(x)
        return embeddings
