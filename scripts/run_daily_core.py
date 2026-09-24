"""Run the corr08 pure-daily Factor Mean or Experiment A validation path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.models.e2eai import E2EAIModel
from e2eai.training.trainer import E2EAITrainer
from e2eai.utils.reproducibility import seed_everything
from e2eai.workflows import load_configured_data, write_metrics


def run(
    config_path: Path,
    output: Path | None = None,
    checkpoint: Path | None = None,
) -> Path:
    config = load_config(config_path)
    if config.experiment_line != "daily_strategy":
        raise ValueError("daily_core requires experiment_line=daily_strategy")
    if list(config.model.horizons) != [2] or list(config.training.optimization_horizons) != [2]:
        raise ValueError("daily_core must use only horizon h=2")
    if config.data.panel_return_horizons:
        raise ValueError("daily_core must not load multi-day panel labels")
    if config.data.test_start is not None or config.data.test_end is not None:
        raise ValueError("daily_core is validation-only and must not configure an OOS split")
    if output is not None:
        config.logging.output_dir = str(output)
    destination = Path(config.logging.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    seed_everything(config.seed)
    source, synthetic = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    split = {**loaders.dataset.split_metadata, "synthetic": synthetic, "oos_evaluated": False}
    write_metrics(destination / "split_metadata.json", split)
    if split.get("used_date_ranges", {}).get("test") is not None:
        raise RuntimeError("daily_core unexpectedly produced a test split")

    model = E2EAIModel(config.model, config.ablation)
    training = config.model.strategy_head == "simple_mlp" and checkpoint is None
    trainer = E2EAITrainer(model, config, create_optimizer=training)
    fit = None
    try:
        if training:
            fit = trainer.fit(
                loaders.train,
                loaders.validation,
                epochs=config.training.epochs,
                output_dir=destination,
            )
            trainer.load_checkpoint(destination / "best.pt", load_optimizer=False, restore_rng=False)
        elif checkpoint is not None:
            trainer.load_checkpoint(checkpoint, load_optimizer=False, restore_rng=False)
        metrics = trainer.evaluate(loaders.validation)
        score_records, score_checksum = trainer.collect_raw_scores(loaders.validation)
    finally:
        trainer.close()

    if fit is not None:
        with (destination / "training.jsonl").open("w", encoding="utf-8") as handle:
            for row in fit.history:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        pd.DataFrame(fit.history).to_csv(
            destination / "epoch_history.csv", index=False, encoding="utf-8-sig"
        )
    else:
        (destination / "training.jsonl").write_text("", encoding="utf-8")
    pd.DataFrame(score_records).to_csv(
        destination / "validation_scores.csv", index=False, encoding="utf-8-sig"
    )

    best_epoch = -1 if fit is None else fit.best_epoch
    experiment_name = "experiment_a" if config.model.strategy_head == "simple_mlp" else "factor_mean"
    result = {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_metric": None if fit is None else fit.best_metric,
        "epochs_completed": 0 if fit is None else len(fit.history),
        "stopped_early": False if fit is None else fit.stopped_early,
        "raw_score_sha256": score_checksum,
        "oos_evaluated": False,
        "daily_core_only": True,
        "horizons_used": [2],
        "multi_day_labels_loaded": False,
        **metrics,
    }
    if fit is not None and fit.history:
        train_rankics = [
            float(row["train_execution_rankic"])
            for row in fit.history
            if row.get("train_execution_rankic") is not None
        ]
        if train_rankics:
            result["train_mean_execution_rankic"] = float(sum(train_rankics) / len(train_rankics))
            result["ic_gap_train_minus_validation"] = float(
                result["train_mean_execution_rankic"]
                - float(metrics.get("validation_execution_rankic", 0.0))
            )
    write_metrics(destination / "validation_metrics.json", result)

    rows: list[dict[str, object]] = []
    for cost in config.evaluation.cost_scenarios_bps:
        token = str(float(cost)).replace(".", "p")
        rows.append(
            {
                "experiment": result["experiment"],
                "best_epoch": best_epoch,
                "cost_bps": float(cost),
                "annual_return": metrics.get(f"validation_daily_return_cost_{token}bps"),
                "sharpe": metrics.get(f"validation_daily_sharpe_cost_{token}bps"),
                "mdd": metrics.get(f"validation_daily_mdd_cost_{token}bps"),
                "turnover": metrics.get("validation_daily_turnover"),
                "effective_n": metrics.get("validation_effective_n"),
                "max_weight": metrics.get("validation_max_weight"),
                "selected_stocks": metrics.get("validation_selected_stock_count"),
                "benchmark_annual_return": metrics.get(
                    f"validation_benchmark_annual_return_cost_{token}bps"
                ),
                "annual_excess_return": metrics.get(
                    f"validation_active_annual_excess_return_cost_{token}bps"
                ),
                "tracking_error": metrics.get(f"validation_tracking_error_cost_{token}bps"),
                "information_ratio": metrics.get(
                    f"validation_information_ratio_cost_{token}bps"
                ),
                "active_mdd": metrics.get(f"validation_active_max_drawdown_cost_{token}bps"),
                "active_win_rate": metrics.get(f"validation_active_win_rate_cost_{token}bps"),
            }
        )
    pd.DataFrame(rows).to_csv(destination / "summary.csv", index=False, encoding="utf-8-sig")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate an existing checkpoint without retraining (used for metric-only revisions).",
    )
    args = parser.parse_args()
    print(run(args.config, args.output, args.checkpoint))


if __name__ == "__main__":
    main()
