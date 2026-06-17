"""
Multimodal Fusion Model Module.
Concatenates embeddings from Audio, Text, MRI, and EHR, and passes them through a classification head.
"""
import torch
import torch.nn as nn

class MultimodalFusionClassifier(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, mri_dim: int, ehr_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super(MultimodalFusionClassifier, self).__init__()
        
        self.total_feature_dim = audio_dim + text_dim + mri_dim + ehr_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor, mri_emb: torch.Tensor, ehr_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_emb: Tensor of shape (batch_size, audio_dim)
            text_emb: Tensor of shape (batch_size, text_dim)
            mri_emb: Tensor of shape (batch_size, mri_dim)
            ehr_emb: Tensor of shape (batch_size, ehr_dim)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Concatenate all embeddings
        fused_features = torch.cat((audio_emb, text_emb, mri_emb, ehr_emb), dim=1)
        
        # Pass through classification head
        logits = self.classifier(fused_features)
        return logits
