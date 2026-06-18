"""
Evaluation Loop.

Runs inference on a DataLoader and computes Accuracy, Precision, Recall,
F1 Score and ROC-AUC using sklearn.  Accepts the same ModuleDict produced
by training.train.build_model().
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.ModuleDict,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Run a full evaluation pass and return classification metrics.

    Parameters
    ----------
    model      : nn.ModuleDict with keys 'audio', 'text', 'mri', 'ehr', 'fusion'.
    dataloader : DataLoader yielding dicts with keys matching MultimodalDataset output.
    device     : torch.device to run inference on.

    Returns
    -------
    metrics : dict with keys accuracy, precision, recall, f1_score, roc_auc.
    """
    model.eval()

    all_preds: list[int] = []
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            # ── Move to device ───────────────────────────────────────────────
            audio = batch["audio"].to(device).squeeze(1)   # (B, T)
            text_ids = batch["text_ids"].to(device)        # (B, seq_len)
            text_mask = batch["text_mask"].to(device)      # (B, seq_len)
            mri = batch["mri"].to(device)                  # (B, 1, D, H, W)
            ehr = batch["ehr"].to(device)                  # (B, n_ehr)
            labels = batch["label"].to(device)             # (B,)

            # ── Forward through each encoder ─────────────────────────────────
            audio_emb = model["audio"](audio)
            text_emb = model["text"](text_ids, text_mask)
            mri_emb = model["mri"](mri)
            ehr_emb = model["ehr"](ehr)

            logits = model["fusion"](audio_emb, text_emb, mri_emb, ehr_emb)

            # ── Collect predictions ──────────────────────────────────────────
            probs = torch.softmax(logits, dim=1)[:, 1]   # Prob of class=1 (MCI)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels, dtype=np.int32)
    y_pred = np.array(all_preds, dtype=np.int32)
    y_prob = np.array(all_probs, dtype=np.float32)

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    logger.info(
        "Evaluation → acc=%.4f | prec=%.4f | rec=%.4f | f1=%.4f | auc=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )

    return metrics
