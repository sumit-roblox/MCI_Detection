"""
Text Encoder Module.
Uses BERT to extract features from transcript text.
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Optional

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", apply_lora: bool = False):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        self.feature_dim = self.bert.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
            attention_mask: Tensor of shape (batch_size, sequence_length)
        Returns:
            Embeddings of shape (batch_size, feature_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Using the [CLS] token representation
        embeddings = outputs.pooler_output
        return embeddings
