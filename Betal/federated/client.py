"""
Flower Federated Learning Client.

Each hospital / clinical site runs one instance of this client.
It loads its own local dataset, trains for a configurable number of
local epochs, and communicates model weights with the central server
without ever sharing raw patient data.

Usage (from project root):
    python -m federated.client --config configs/config.yaml --client-id 0
"""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
import flwr as fl
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader

from datasets.multimodal_dataset import MultimodalDataset
from training.train import build_model, set_seed, train_one_epoch
from training.evaluate import evaluate_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Helper: parameter serialisation ─────────────────────────────────────────

def get_parameters(model: nn.ModuleDict) -> NDArrays:
    """Extract all model parameters as a flat list of numpy arrays."""
    return [
        val.cpu().detach().numpy()
        for _, module in model.items()
        for val in module.state_dict().values()
    ]


def set_parameters(model: nn.ModuleDict, parameters: NDArrays) -> None:
    """Load a flat list of numpy arrays back into the model in-place."""
    param_idx = 0
    for _, module in model.items():
        keys = list(module.state_dict().keys())
        new_state = OrderedDict()
        for key in keys:
            new_state[key] = torch.from_numpy(np.copy(parameters[param_idx]))
            param_idx += 1
        module.load_state_dict(new_state, strict=True)


# ── Flower client ────────────────────────────────────────────────────────────

class MultimodalFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient wrapping the multimodal training pipeline."""

    def __init__(
        self,
        model: nn.ModuleDict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=1e-4,
        )

    # ── Flower interface ─────────────────────────────────────────────────────

    def get_parameters(self, config: dict) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: dict
    ) -> Tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Receive global parameters → run local training → return updated parameters.
        """
        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", 1))
        total_loss = 0.0

        for epoch in range(local_epochs):
            loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )
            total_loss += loss
            logger.info(f"[Client] local epoch {epoch + 1}/{local_epochs} | loss={loss:.4f}")

        avg_loss = total_loss / max(local_epochs, 1)

        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {"train_loss": float(avg_loss)},
        )

    def evaluate(
        self, parameters: NDArrays, config: dict
    ) -> Tuple[float, int, dict[str, Scalar]]:
        """
        Receive global parameters → evaluate on local held-out set → return loss & metrics.
        """
        set_parameters(self.model, parameters)

        metrics = evaluate_model(self.model, self.val_loader, self.device)

        # Compute cross-entropy loss on val set for Flower compatibility
        self.model.eval()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.val_loader:
                audio = batch["audio"].to(self.device).squeeze(1)
                text_ids = batch["text_ids"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                mri = batch["mri"].to(self.device)
                ehr = batch["ehr"].to(self.device)
                labels = batch["label"].to(self.device)

                audio_emb = self.model["audio"](audio)
                text_emb = self.model["text"](text_ids, text_mask)
                mri_emb = self.model["mri"](mri)
                ehr_emb = self.model["ehr"](ehr)
                logits = self.model["fusion"](audio_emb, text_emb, mri_emb, ehr_emb)

                loss = criterion(logits, labels)
                total_loss += loss.item()

        avg_loss = total_loss / max(len(self.val_loader), 1)

        return (
            float(avg_loss),
            len(self.val_loader.dataset),
            {
                "accuracy": float(metrics["accuracy"]),
                "f1_score": float(metrics["f1_score"]),
                "roc_auc": float(metrics["roc_auc"]),
            },
        )


# ── Entry point ──────────────────────────────────────────────────────────────

def start_client(cfg: dict, client_id: int = 0) -> None:
    """
    Instantiate model + data for a given client and connect to the FL server.

    In a real multi-site deployment each client reads from its own local
    metadata CSV pointing to that site's data.  Here we use the same CSV
    with a simulated client_id partition.
    """
    set_seed(42 + client_id)  # Different seed per client for data diversity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build dataset ────────────────────────────────────────────────────────
    train_ds = MultimodalDataset(
        metadata_path=cfg["data"]["metadata_path"],
        data_dir=cfg["data"]["data_dir"],
        text_model_name=cfg["model"]["text_model_name"],
        is_train=True,
    )
    val_ds = MultimodalDataset(
        metadata_path=cfg["data"]["metadata_path"],
        data_dir=cfg["data"]["data_dir"],
        text_model_name=cfg["model"]["text_model_name"],
        is_train=False,
        preprocessor=train_ds.preprocessor,
    )

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model(cfg, device)

    # ── Start Flower client ──────────────────────────────────────────────────
    flower_client = MultimodalFlowerClient(model, train_loader, val_loader, cfg, device)
    server_address = cfg["federated"]["server_address"]

    logger.info(f"[Client {client_id}] Connecting to {server_address}")
    fl.client.start_numpy_client(server_address=server_address, client=flower_client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a Flower federated client.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--client-id", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    start_client(config, client_id=args.client_id)
