"""
Main Training Orchestrator.

Wires together all four encoders + fusion classifier for end-to-end
local training.  Also exposed as a library function so the federated
client can call it directly.

Run standalone:
    python -m training.train --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from datasets.multimodal_dataset import MultimodalDataset
from models.audio_encoder import AudioEncoder
from models.ehr_encoder import EHREncoder
from models.fusion_model import MultimodalFusionClassifier
from models.mri_encoder import MRIEncoder
from models.text_encoder import TextEncoder
from training.evaluate import evaluate_model
from training.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model factory ────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device) -> nn.ModuleDict:
    """
    Instantiate all modality encoders and the fusion head.

    Returns a ModuleDict so that:
      - All sub-models are registered as proper parameters.
      - Individual encoders can be addressed by name (for LoRA injection).
    """
    audio_enc = AudioEncoder(model_name=cfg["model"]["audio_model_name"])
    text_enc = TextEncoder(model_name=cfg["model"]["text_model_name"])
    mri_enc = MRIEncoder(
        in_channels=cfg["model"]["mri_in_channels"],
        out_features=cfg["model"]["mri_out_features"],
    )
    ehr_enc = EHREncoder(
        in_features=cfg["model"]["ehr_in_features"],
        out_features=cfg["model"]["ehr_out_features"],
    )
    fusion = MultimodalFusionClassifier(
        audio_dim=audio_enc.feature_dim,
        text_dim=text_enc.feature_dim,
        mri_dim=mri_enc.feature_dim,
        ehr_dim=ehr_enc.feature_dim,
        hidden_dim=cfg["model"]["fusion_hidden_dim"],
        num_classes=cfg["model"]["num_classes"],
    )

    model = nn.ModuleDict({
        "audio": audio_enc,
        "text": text_enc,
        "mri": mri_enc,
        "ehr": ehr_enc,
        "fusion": fusion,
    })
    return model.to(device)


# ── Forward pass helper ──────────────────────────────────────────────────────

def forward_pass(
    model: nn.ModuleDict,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass through all encoders + fusion head.

    Returns
    -------
    logits : (B, num_classes)
    labels : (B,)
    """
    audio = batch["audio"].to(device)             # (B, 1, T)
    text_ids = batch["text_ids"].to(device)       # (B, seq_len)
    text_mask = batch["text_mask"].to(device)     # (B, seq_len)
    mri = batch["mri"].to(device)                 # (B, 1, D, H, W)
    ehr = batch["ehr"].to(device)                 # (B, n_ehr)
    labels = batch["label"].to(device)            # (B,)

    # Squeeze channel dim expected by Wav2Vec2  →  (B, T)
    audio_input = audio.squeeze(1)

    audio_emb = model["audio"](audio_input)
    text_emb = model["text"](text_ids, text_mask)
    mri_emb = model["mri"](mri)
    ehr_emb = model["ehr"](ehr)

    logits = model["fusion"](audio_emb, text_emb, mri_emb, ehr_emb)
    return logits, labels


# ── Training epoch ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.ModuleDict,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    """
    Run one full training epoch.

    Returns
    -------
    mean_loss : float  — average cross-entropy loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        optimizer.zero_grad()

        logits, labels = forward_pass(model, batch, device)
        loss = criterion(logits, labels)

        loss.backward()
        # Gradient clipping prevents exploding gradients from large transformers
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            logger.info(f"  Step {step:04d} | loss={loss.item():.4f}")

    return total_loss / max(len(loader), 1)


# ── Full training pipeline ───────────────────────────────────────────────────

def run_training(cfg: dict) -> nn.ModuleDict:
    """
    Full training + validation pipeline.

    Parameters
    ----------
    cfg : dict  — Parsed config.yaml content.

    Returns
    -------
    Trained model (nn.ModuleDict) on CPU.
    """
    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Dataset & DataLoaders ────────────────────────────────────────────────
    full_ds = MultimodalDataset(
        metadata_path=cfg["data"]["metadata_path"],
        data_dir=cfg["data"]["data_dir"],
        text_model_name=cfg["model"]["text_model_name"],
        is_train=True,
    )

    n_total = len(full_ds)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val

    train_ds, val_ds_raw = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Validation set uses the training preprocessor — no leakage
    val_ds = MultimodalDataset(
        metadata_path=cfg["data"]["metadata_path"],
        data_dir=cfg["data"]["data_dir"],
        text_model_name=cfg["model"]["text_model_name"],
        is_train=False,
        preprocessor=full_ds.preprocessor,
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train samples: {n_train} | Val samples: {n_val}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg, device)

    # ── Optim & Scheduler ────────────────────────────────────────────────────
    lr = cfg["training"]["learning_rate"]
    epochs = cfg["training"]["epochs"]

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_ckpt_path = Path("checkpoints/best_model.pt")
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"  Train loss : {train_loss:.4f}\n"
            f"  Val metrics: acc={val_metrics['accuracy']:.4f} | "
            f"f1={val_metrics['f1_score']:.4f} | "
            f"auc={val_metrics['roc_auc']:.4f}"
        )

        # Save best checkpoint by F1 score
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": {k: v.state_dict() for k, v in model.items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
            logger.info(f"  ✔ New best model saved (F1={best_f1:.4f})")

    logger.info("\nTraining complete.")
    return model.cpu()


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal MCI classifier.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_training(config)
