"""
Audio Encoder Module.
Uses Wav2Vec2 to extract features from raw speech audio.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from typing import Optional, Dict, Any

class AudioEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/wav2vec2-base", apply_lora: bool = False):
        super(AudioEncoder, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Note: If apply_lora is True, LoRA configuration is handled in lora_setup.py via PEFT.
            
        self.feature_dim = self.wav2vec2.config.hidden_size

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_values: Tensor of shape (batch_size, sequence_length)
            attention_mask: Tensor of shape (batch_size, sequence_length)
        Returns:
            Embeddings of shape (batch_size, feature_dim)
        """
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        # Using the mean over the sequence length as the aggregate embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
