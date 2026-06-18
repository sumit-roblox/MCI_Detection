"""
Inference / Prediction Script.

Loads a saved checkpoint and runs inference on a single patient's data
across all four modalities.

Usage:
    python predict.py \\
        --config configs/config.yaml \\
        --checkpoint checkpoints/best_model.pt \\
        --audio data/audio/patient_001.wav \\
        --text data/transcripts/patient_001.txt \\
        --mri data/mri/patient_001.nii \\
        --ehr 75,1,24.5,0,1,120,80    # comma-separated EHR feature values
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from training.train import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Healthy", 1: "Mild Cognitive Impairment (MCI)"}


def load_model_from_checkpoint(cfg: dict, ckpt_path: str, device: torch.device) -> torch.nn.ModuleDict:
    """Load model weights from a training checkpoint."""
    model = build_model(cfg, device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dicts = checkpoint["model_state_dict"]
    for key, module in model.items():
        module.load_state_dict(state_dicts[key])
    model.eval()
    logger.info(f"Checkpoint loaded from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


def predict(cfg: dict, ckpt_path: str, audio_path: str, text_path: str, mri_path: str, ehr_values: str) -> None:
    """Run inference for a single patient across all modalities."""
    import torchaudio
    import nibabel as nib
    from transformers import BertTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    # ── Audio ──────────────────────────────────────────────────────────────
    waveform, sr = torchaudio.load(audio_path)
    target_sr = cfg["model"]["target_sr"]
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_tensor = waveform.squeeze(0).unsqueeze(0).to(device)  # (1, T)

    # ── Text ───────────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(cfg["model"]["text_model_name"])
    text = Path(text_path).read_text(encoding="utf-8").strip()
    enc = tokenizer(text, max_length=cfg["model"]["max_text_len"], padding="max_length",
                    truncation=True, return_tensors="pt")
    text_ids = enc["input_ids"].to(device)
    text_mask = enc["attention_mask"].to(device)

    # ── MRI ────────────────────────────────────────────────────────────────
    from datasets.multimodal_dataset import MultimodalDataset
    img = nib.load(mri_path)
    volume = img.get_fdata(dtype=np.float32)
    v_min, v_max = volume.min(), volume.max()
    if v_max > v_min:
        volume = (volume - v_min) / (v_max - v_min)
    target_shape = tuple(cfg["model"]["mri_target_shape"])
    volume = MultimodalDataset._crop_or_pad_volume(volume, target_shape)
    mri_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)

    # ── EHR ────────────────────────────────────────────────────────────────
    ehr_vals = np.array([float(v) for v in ehr_values.split(",")], dtype=np.float32)
    ehr_tensor = torch.from_numpy(ehr_vals).unsqueeze(0).to(device)  # (1, n_features)

    # ── Inference ──────────────────────────────────────────────────────────
    with torch.no_grad():
        audio_emb = model["audio"](audio_tensor)
        text_emb = model["text"](text_ids, text_mask)
        mri_emb = model["mri"](mri_tensor)
        ehr_emb = model["ehr"](ehr_tensor)
        logits = model["fusion"](audio_emb, text_emb, mri_emb, ehr_emb)

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_class = int(probs.argmax())

    logger.info("\n" + "=" * 50)
    logger.info(f"Prediction     : {CLASS_NAMES[pred_class]}")
    logger.info(f"Confidence     : {probs[pred_class] * 100:.1f}%")
    logger.info(f"  P(Healthy)   : {probs[0] * 100:.1f}%")
    logger.info(f"  P(MCI)       : {probs[1] * 100:.1f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal MCI prediction.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--mri", required=True)
    parser.add_argument("--ehr", required=True, help="Comma-separated EHR feature values")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    predict(config, args.checkpoint, args.audio, args.text, args.mri, args.ehr)
