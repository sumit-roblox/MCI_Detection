"""
LoRA Fine-tuning Setup.
Applies LoRA to BERT and Wav2Vec2 models to train only adapter layers.
"""
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

def apply_lora_to_text(model: nn.Module, r: int = 8, alpha: int = 16) -> nn.Module:
    """
    Apply LoRA to the BERT text encoder.
    """
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        r=r, 
        lora_alpha=alpha, 
        target_modules=["query", "value"],
        lora_dropout=0.1
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model

def apply_lora_to_audio(model: nn.Module, r: int = 8, alpha: int = 16) -> nn.Module:
    """
    Apply LoRA to the Wav2Vec2 audio encoder.
    """
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        r=r, 
        lora_alpha=alpha, 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
