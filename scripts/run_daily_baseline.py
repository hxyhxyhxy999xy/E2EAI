"""Evaluate the non-learned equal-factor Top20% baseline on validation only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.models.e2eai import E2EAIModel
from e2eai.training.trainer import E2EAITrainer
from e2eai.utils.reproducibility import seed_everything
from e2eai.workflows import load_configured_data, write_metrics
from scripts.run_daily_experiment import _apply_fold, _key_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baselines/factor_mean_top20.yaml")
    )
    parser.add_argument(
        "--fold-config", type=Path, default=Path("configs/rolling_fold0_2018_2022.yaml")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    _apply_fold(config, args.fold_config, 0)
    if args.output is not None:
        config.logging.output_dir = str(args.output)
    destination = Path(config.logging.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_metrics(destination / "key_parameters.json", _key_parameters(config))
    seed_everything(config.seed)
    source, _ = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    model = E2EAIModel(config.model, config.ablation)
    trainer = E2EAITrainer(model, config)
    try:
        metrics = trainer.evaluate(loaders.validation)
        score_records, score_checksum = trainer.collect_raw_scores(loaders.validation)
    finally:
        trainer.close()
    metrics["oos_evaluated"] = False
    metrics["raw_score_sha256"] = score_checksum
    write_metrics(destination / "validation_metrics.json", metrics)
    pd.DataFrame(score_records).to_csv(
        destination / "validation_scores.csv", index=False, encoding="utf-8-sig"
    )
    rows = []
    for cost in config.evaluation.cost_scenarios_bps:
        token = str(float(cost)).replace(".", "p")
        rows.append({
            "experiment": "baseline_factor_mean_top20",
            "best_epoch": None,
            "daily_return": metrics.get(f"validation_daily_return_cost_{token}bps"),
            "daily_sharpe": metrics.get(f"validation_daily_sharpe_cost_{token}bps"),
            "mdd": metrics.get(f"validation_daily_mdd_cost_{token}bps"),
            "turnover": metrics.get("validation_daily_turnover"),
            "effective_n": metrics.get("validation_effective_n"),
            "max_weight": metrics.get("validation_max_weight"),
            "selected_stocks": metrics.get("validation_selected_stock_count"),
            "cost_bps": float(cost),
        })
    pd.DataFrame(rows).to_csv(destination / "summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
