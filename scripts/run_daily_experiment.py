"""Train one configured daily-strategy experiment without touching the OOS split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch
import yaml

from e2eai.config import ExperimentConfig, load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.models.e2eai import E2EAIModel
from e2eai.training.trainer import E2EAITrainer
from e2eai.utils.reproducibility import seed_everything
from e2eai.workflows import load_configured_data, write_metrics


def _apply_fold(config: ExperimentConfig, fold_path: Path, fold_id: int) -> None:
    values = yaml.safe_load(fold_path.read_text(encoding="utf-8")) or {}
    folds = values.get("folds", [])
    matches = [fold for fold in folds if int(fold.get("id", -1)) == fold_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one fold id={fold_id} in {fold_path}")
    fold = matches[0]
    config.data.train_start, config.data.train_end = fold["train"]
    config.data.validation_start, config.data.validation_end = fold["validation"]
    config.data.test_start, config.data.test_end = fold["oos"]


def _key_parameters(config: ExperimentConfig) -> dict[str, object]:
    return {
        "experiment_line": config.experiment_line,
        "universe": config.data.index_universe,
        "horizons": config.model.horizons,
        "execution_horizon": config.model.execution_horizon,
        "factor_selection_pooling": config.model.factor_selection.pooling,
        "gamma_f": config.model.factor_selection.gamma_f,
        "gate_mode_train": config.model.factor_selection.gate_mode_train,
        "gate_mode_eval": config.model.factor_selection.gate_mode_eval,
        "direction_normalization": config.model.direction.normalization,
        "direction_reset_each_epoch": config.model.direction.reset_each_epoch,
        "gat_heads": config.model.gat.heads,
        "add_self_loops": config.model.gat.add_self_loops,
        "universe_graph_mode": config.data.universe_graph_mode,
        "allocation_mode": config.model.portfolio.allocation_mode,
        "gamma_p": config.model.portfolio.gamma_p,
        "gamma_p_mode": config.model.portfolio.gamma_p_mode,
        "theta": config.model.portfolio.theta,
        "lambda_portfolio": config.loss.lambda_portfolio,
        "lambda_s": config.loss.lambda_s,
        "lambda_f": config.loss.lambda_f,
        "lambda_e": config.loss.lambda_e,
        "lambda_score": config.loss.lambda_score,
        "batch_dates": config.training.batch_dates,
        "lr": config.global_optimizer.lr,
        "epochs": config.training.epochs,
        "patience": config.training.early_stopping_patience,
        "validation_target": config.training.validation_target,
        "seed": config.seed,
        "ablation": config.ablation.__dict__,
        "experiment": config.experiment.__dict__,
    }


def _normalized_score_config(config: ExperimentConfig) -> dict[str, object]:
    """Return every score-affecting setting; portfolio construction is excluded."""
    values = copy.deepcopy(config.to_dict())
    values.pop("logging", None)
    values.pop("experiment", None)
    model = values.get("model")
    if isinstance(model, dict):
        model.pop("portfolio", None)
    return values


def assert_evaluation_only_compatible(
    source: ExperimentConfig,
    evaluation: ExperimentConfig,
) -> None:
    """Reject any B-vs-A change outside portfolio construction and run metadata."""
    if evaluation.experiment.mode != "evaluation_only":
        raise ValueError("The comparison experiment must be evaluation_only")
    if _normalized_score_config(source) != _normalized_score_config(evaluation):
        raise ValueError(
            "evaluation_only experiment differs from its source outside portfolio settings"
        )


def _model_state_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _prepare_synthetic_smoke(config: ExperimentConfig) -> None:
    """Use a small in-memory split while preserving the configured model structure."""
    config.device = "cpu"
    config.data.path = None
    config.data.return_path = None
    config.data.execution_return_path = None
    config.data.benchmark_path = None
    config.data.membership_path = None
    config.data.industry_npz_path = None
    config.data.index_universe = None
    config.data.train_start = None
    config.data.train_end = None
    config.data.validation_start = None
    config.data.validation_end = None
    config.data.test_start = None
    config.data.test_end = None
    config.data.purge_label_overlap = False
    config.data.train_ratio = 0.50
    config.data.validation_ratio = 0.25
    config.data.test_ratio = 0.25
    config.synthetic.num_factors = config.model.num_factors
    config.synthetic.num_dates = 20
    minimum_assets = (
        math.ceil(1.0 / config.model.portfolio.theta)
        if config.model.portfolio.allocation_mode == "capped_simplex"
        else 1
    )
    config.synthetic.num_stocks = max(80, minimum_assets)
    config.synthetic.num_industries = 8
    config.synthetic.active_probability = 1.0
    config.training.batch_dates = 2
    config.training.epochs = 1
    config.training.early_stopping_patience = 2
    config.training.num_workers = 0
    config.training.mixed_precision = False
    config.logging.tensorboard = False
    config.logging.wandb = False


def _select_factor_candidate(
    config: ExperimentConfig,
    config_path: Path,
    destination: Path,
    fold_path: Path,
    fold_id: int,
    synthetic_smoke: bool,
) -> str | None:
    candidates = config.experiment.selection_candidates
    if not candidates:
        return None
    scored: list[tuple[float, str]] = []
    for name in candidates:
        metrics_path = destination.parent / name / "validation_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing candidate metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = float(metrics["best_metric"])
        if not math.isfinite(value):
            raise ValueError(f"Candidate {name} has a non-finite best_metric")
        scored.append((value, name))
    selected = max(scored, key=lambda item: item[0])[1]
    selected_path = config_path.with_name(f"{selected}.yaml")
    selected_config = load_config(selected_path)
    if synthetic_smoke:
        _prepare_synthetic_smoke(selected_config)
    else:
        _apply_fold(selected_config, fold_path, fold_id)
    candidate_values = selected_config.to_dict()
    target_values = config.to_dict()
    candidate_values.pop("logging", None)
    target_values.pop("logging", None)
    candidate_values.pop("experiment", None)
    target_values.pop("experiment", None)
    candidate_values["ablation"]["use_industry_gat"] = True
    candidate_values["model"]["factor_selection"] = target_values["model"][
        "factor_selection"
    ]
    if candidate_values != target_values:
        raise ValueError("Experiment D must differ from its selected C only by Industry GAT")
    config.model.factor_selection = copy.deepcopy(selected_config.model.factor_selection)
    return selected


def run(
    config_path: Path,
    fold_path: Path,
    fold_id: int,
    output: Path | None,
    *,
    synthetic_smoke: bool = False,
) -> Path:
    config = load_config(config_path)
    if config.experiment_line != "daily_strategy":
        raise ValueError("run_daily_experiment requires experiment_line=daily_strategy")
    if synthetic_smoke:
        _prepare_synthetic_smoke(config)
    else:
        _apply_fold(config, fold_path, fold_id)
    if output is not None:
        config.logging.output_dir = str(output)
    destination = Path(config.logging.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_candidate = _select_factor_candidate(
        config,
        config_path,
        destination,
        fold_path,
        fold_id,
        synthetic_smoke,
    )
    config.validate()

    resolved = config.to_dict()
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    key_parameters = _key_parameters(config)
    print(json.dumps(key_parameters, ensure_ascii=False, indent=2), flush=True)
    write_metrics(destination / "key_parameters.json", key_parameters)

    seed_everything(config.seed)
    source, synthetic = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    write_metrics(
        destination / "split_metadata.json",
        {**loaders.dataset.split_metadata, "synthetic": synthetic, "oos_evaluated": False},
    )
    evaluation_only = config.experiment.mode == "evaluation_only"
    source_experiment = config.experiment.source_experiment
    source_checkpoint: Path | None = None
    source_score_metadata: dict[str, object] | None = None
    if evaluation_only:
        assert source_experiment is not None
        source_config = load_config(config_path.with_name(f"{source_experiment}.yaml"))
        if synthetic_smoke:
            _prepare_synthetic_smoke(source_config)
        else:
            _apply_fold(source_config, fold_path, fold_id)
        assert_evaluation_only_compatible(source_config, config)
        source_directory = destination.parent / source_experiment
        source_checkpoint = source_directory / "best.pt"
        metadata_path = source_directory / "score_metadata.json"
        if not source_checkpoint.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "evaluation_only requires the source best.pt and score_metadata.json"
            )
        source_score_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    model = E2EAIModel(config.model, config.ablation)
    trainer = E2EAITrainer(model, config, create_optimizer=not evaluation_only)
    try:
        if evaluation_only:
            assert source_checkpoint is not None
            checkpoint_payload = trainer.load_checkpoint(
                source_checkpoint, load_optimizer=False, restore_rng=False
            )
            fit = None
        else:
            fit = trainer.fit(
                loaders.train,
                loaders.validation,
                epochs=config.training.epochs,
                output_dir=destination,
            )
            checkpoint_payload = trainer.load_checkpoint(
                destination / "best.pt", load_optimizer=False, restore_rng=False
            )
        validation = trainer.evaluate(loaders.validation)
        score_records, score_checksum = trainer.collect_raw_scores(loaders.validation)
        parameter_checksum = _model_state_checksum(model)
    finally:
        trainer.close()

    with (destination / "training.jsonl").open("w", encoding="utf-8") as handle:
        if fit is not None:
            for row in fit.history:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    pd.DataFrame(score_records).to_csv(
        destination / "validation_scores.csv", index=False, encoding="utf-8-sig"
    )
    if source_score_metadata is not None:
        if score_checksum != source_score_metadata.get("raw_score_sha256"):
            raise RuntimeError("Evaluation-only raw scores differ from the source experiment")
        if parameter_checksum != source_score_metadata.get("model_state_sha256"):
            raise RuntimeError("Evaluation-only model parameters differ from the source checkpoint")
    score_metadata = {
        "trained_from": source_experiment if evaluation_only else config_path.stem,
        "evaluation_only": evaluation_only,
        "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
        "model_state_sha256": parameter_checksum,
        "raw_score_sha256": score_checksum,
        "score_rows": len(score_records),
        "selected_factor_candidate": selected_candidate,
        "source_checkpoint_best_metric": (
            checkpoint_payload.get("best_validation_metric")
            if evaluation_only
            else None
        ),
    }
    write_metrics(destination / "score_metadata.json", score_metadata)
    best_epoch = (
        int(checkpoint_payload.get("best_epoch", checkpoint_payload.get("epoch", -1)))
        if fit is None
        else fit.best_epoch
    )
    best_metric = (
        validation[config.training.validation_target]
        if fit is None
        else fit.best_metric
    )
    result = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "epochs_completed": 0 if fit is None else len(fit.history),
        "stopped_early": False if fit is None else fit.stopped_early,
        **score_metadata,
        "oos_evaluated": False,
        **validation,
    }
    write_metrics(destination / "validation_metrics.json", result)
    summary_rows = []
    for cost in config.evaluation.cost_scenarios_bps:
        token = str(float(cost)).replace(".", "p")
        summary_rows.append(
            {
                "experiment": config_path.stem,
                "best_epoch": best_epoch,
                "evaluation_only": evaluation_only,
                "raw_score_sha256": score_checksum,
                "daily_return": validation.get(f"validation_daily_return_cost_{token}bps"),
                "daily_sharpe": validation.get(f"validation_daily_sharpe_cost_{token}bps"),
                "mdd": validation.get(f"validation_daily_mdd_cost_{token}bps"),
                "turnover": validation.get("validation_daily_turnover"),
                "effective_n": validation.get("validation_effective_n"),
                "max_weight": validation.get("validation_max_weight"),
                "selected_stocks": validation.get("validation_selected_stock_count"),
                "cost_bps": float(cost),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        destination / "summary.csv", index=False, encoding="utf-8-sig"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--fold-config",
        type=Path,
        default=Path("configs/rolling_fold0_2018_2022.yaml"),
    )
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    print(
        run(
            args.config,
            args.fold_config,
            args.fold_id,
            args.output,
            synthetic_smoke=args.synthetic_smoke,
        )
    )


if __name__ == "__main__":
    main()
