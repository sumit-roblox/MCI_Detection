"""
Flower Federated Learning Server.

Runs the global aggregation loop using FedAvg. Logs per-round metrics
aggregated from all participating clients.

Usage (from project root):
    python -m federated.server --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import flwr as fl
import yaml
from flwr.common import Metrics
from flwr.server.strategy import FedAvg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Custom metric aggregation ────────────────────────────────────────────────

def weighted_average_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    Weighted average of client evaluation metrics.

    Flower passes a list of (num_examples, metric_dict) tuples.
    We compute a dataset-size-weighted average for each metric key.
    """
    if not metrics:
        return {}

    total_examples = sum(n for n, _ in metrics)
    aggregated: dict[str, float] = {}

    # Collect all metric keys from the first client
    metric_keys = list(metrics[0][1].keys())

    for key in metric_keys:
        aggregated[key] = (
            sum(n * m.get(key, 0.0) for n, m in metrics) / total_examples
        )

    logger.info(
        "Aggregated eval metrics: "
        + " | ".join(f"{k}={v:.4f}" for k, v in aggregated.items())
    )
    return aggregated


def weighted_average_fit_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Weighted average of client fit metrics (e.g. train_loss)."""
    if not metrics:
        return {}

    total_examples = sum(n for n, _ in metrics)
    metric_keys = list(metrics[0][1].keys())
    aggregated: dict[str, float] = {}

    for key in metric_keys:
        aggregated[key] = (
            sum(n * m.get(key, 0.0) for n, m in metrics) / total_examples
        )

    logger.info(
        "Aggregated fit metrics: "
        + " | ".join(f"{k}={v:.4f}" for k, v in aggregated.items())
    )
    return aggregated


# ── Strategy factory ─────────────────────────────────────────────────────────

def build_fedavg_strategy(cfg: dict) -> FedAvg:
    """
    Build a FedAvg strategy from config.yaml federated settings.

    Key design choices:
    - fraction_fit=1.0   — all available clients train each round (suitable for
                           small N; reduce for large deployments).
    - evaluate_metrics_aggregation_fn — logs per-round val metrics on the server.
    - fit_metrics_aggregation_fn       — logs per-round train loss on the server.
    """
    fed_cfg = cfg.get("federated", {})

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=fed_cfg.get("min_clients", 2),
        min_evaluate_clients=fed_cfg.get("min_clients", 2),
        min_available_clients=fed_cfg.get("min_clients", 2),
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        fit_metrics_aggregation_fn=weighted_average_fit_metrics,
        # Pass local_epochs to clients via config
        on_fit_config_fn=lambda rnd: {"local_epochs": fed_cfg.get("local_epochs", 1)},
    )
    return strategy


# ── Entry point ──────────────────────────────────────────────────────────────

def start_server(cfg: dict) -> None:
    """Start the Flower federated server with FedAvg aggregation."""
    server_address = cfg["federated"]["server_address"]
    num_rounds = cfg["federated"]["num_rounds"]

    strategy = build_fedavg_strategy(cfg)

    logger.info(f"Starting federated server at {server_address} for {num_rounds} rounds...")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Flower federated server.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    start_server(config)
