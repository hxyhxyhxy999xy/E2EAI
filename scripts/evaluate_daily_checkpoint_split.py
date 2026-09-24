"""Evaluate a saved daily-strategy checkpoint on one non-OOS split."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.models.e2eai import E2EAIModel
from e2eai.training.trainer import E2EAITrainer
from e2eai.utils.reproducibility import seed_everything
from e2eai.workflows import load_configured_data, write_metrics
from scripts.run_daily_experiment import _apply_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.experiment_line != "daily_strategy":
        raise ValueError("Only daily_strategy checkpoints are supported")
    _apply_fold(config, args.fold_config, 0)
    seed_everything(config.seed)
    source, _ = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    loader = loaders.train if args.split == "train" else loaders.validation
    model = E2EAIModel(config.model, config.ablation)
    trainer = E2EAITrainer(model, config, create_optimizer=False)
    try:
        payload = trainer.load_checkpoint(args.checkpoint, load_optimizer=False, restore_rng=False)
        metrics = trainer.evaluate(loader)
    finally:
        trainer.close()
    metrics.update(
        split=args.split,
        checkpoint=str(args.checkpoint),
        checkpoint_best_epoch=int(payload.get("best_epoch", payload.get("epoch", -1))),
        oos_evaluated=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_metrics(args.output, metrics)
    (args.output.with_suffix(".yaml")).write_text(
        yaml.safe_dump({"split": args.split, "oos_evaluated": False}, sort_keys=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
