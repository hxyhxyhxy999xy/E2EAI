"""Create a readable convergence audit from batch and validation JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--train-dates", type=int, required=True)
    parser.add_argument("--batch-dates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.history.read_text(encoding="utf-8").splitlines()]
    train = [row for row in rows if "train/total_loss" in row]
    validation = [row for row in rows if "validation/validation_loss" in row]
    batches = math.ceil(args.train_dates / args.batch_dates)
    epochs = np.arange(len(validation))

    def train_epoch_mean(key: str) -> np.ndarray:
        values = np.asarray([row[key] for row in train], dtype=float)
        return np.asarray([np.mean(values[i * batches:(i + 1) * batches]) for i in epochs])

    validation_loss = np.asarray([row["validation/validation_loss"] for row in validation])
    best = int(np.argmin(validation_loss))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(epochs, train_epoch_mean("train/total_loss"), label="Train total loss", linewidth=2)
    axes[0].plot(epochs, validation_loss, label="Validation total loss", linewidth=2)
    axes[0].axvline(best, color="black", linestyle="--", linewidth=1.2,
                    label=f"Best validation epoch = {best}")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CSI500 Fold 0: total loss and early stopping")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.25)

    component_keys = [
        ("validation/portfolio_loss", "Portfolio"),
        ("validation/stability_loss", "Stability (unweighted)"),
        ("validation/factor_return_loss", "Factor return (unweighted)"),
        ("validation/attention_estimate_loss", "Approximation (unweighted)"),
    ]
    for key, label in component_keys:
        axes[1].plot(epochs, [row[key] for row in validation], label=label, linewidth=1.8)
    axes[1].axvline(best, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation component")
    axes[1].set_title("Validation loss components")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
