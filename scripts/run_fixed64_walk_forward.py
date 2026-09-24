"""Run the fixed-64 annual walk-forward baseline fold pipeline.

The module deliberately reuses the accepted daily A/C/D-Stability components,
but makes every train/validation/OOS boundary explicit and fold-local.  It
does not select factors, tune temperature, or inspect an OOS result while
choosing a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2eai.config import ExperimentConfig, load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.evaluation.continuous_portfolio import continuous_long_only_weights
from e2eai.evaluation.daily_validation import _simulate_daily_weight_path
from e2eai.evaluation.metrics import maximum_drawdown
from e2eai.models.e2eai import E2EAIModel
from e2eai.training.trainer import E2EAITrainer
from e2eai.utils.reproducibility import resolve_device, seed_everything
from e2eai.workflows import load_configured_data, write_metrics
from scripts.run_daily_experiment_b import _benchmark_for_dates, _build_weights, _canonical_dates, _performance, _sha256
from scripts.run_daily_experiment_c import _collect_scores, _evaluate, _mlp_parameters, _raw_scores, _score_loss, _gross_loss, _rank_corr, _move_batch
from scripts.run_daily_experiment_d import _batch_specs, _collate_rows, _net_components
from e2eai.training.net_return import numpy_drifted_rows


LAMBDA_PORTFOLIO = 70.2102429073
COST_BPS = 7.5
TEMPERATURE = 1.0
STABILITY_LR = 1e-4
FOLDS: dict[int, dict[str, str]] = {
    0: {"train_start": "2018-01-01", "train_end": "2020-12-31", "validation_start": "2021-01-01", "validation_end": "2021-12-31", "oos_start": "2022-01-01", "oos_end": "2022-12-31"},
    1: {"train_start": "2019-01-01", "train_end": "2021-12-31", "validation_start": "2022-01-01", "validation_end": "2022-12-31", "oos_start": "2023-01-01", "oos_end": "2023-12-31"},
    2: {"train_start": "2020-01-01", "train_end": "2022-12-31", "validation_start": "2023-01-01", "validation_end": "2023-12-31", "oos_start": "2024-01-01", "oos_end": "2024-12-31"},
    3: {"train_start": "2021-01-01", "train_end": "2023-12-31", "validation_start": "2024-01-01", "validation_end": "2024-12-31", "oos_start": "2025-01-01", "oos_end": "2025-12-31"},
    4: {"train_start": "2022-01-01", "train_end": "2024-12-31", "validation_start": "2025-01-01", "validation_end": "2025-12-31", "oos_start": "2026-01-01", "oos_end": "9999-12-31"},
}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _sha(path: Path) -> str:
    return _sha256(path)


def _canonical(values: Iterable[Any]) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in values]


def _root(output: Path) -> Path:
    return output.resolve()


def _preflight_path(output: Path) -> Path:
    return _root(output) / "data_preflight" / "preflight.json"


def _load_metadata(config: ExperimentConfig) -> dict[str, Any]:
    path = Path(config.data.path or "") / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"fixed panel metadata missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_factor_audit(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    list_path = Path(config.data.factor_selection_manifest_path or "").parent / "factor_list.json"
    direction_path = Path(config.data.factor_directions_path or "")
    if not list_path.is_file() or not direction_path.is_file():
        raise FileNotFoundError("fixed factor artifacts are not available")
    factor_values = json.loads(list_path.read_text(encoding="utf-8"))
    names = factor_values.get("factor_names", factor_values) if isinstance(factor_values, dict) else factor_values
    directions_raw = json.loads(direction_path.read_text(encoding="utf-8"))
    directions = directions_raw.get("directions", directions_raw) if isinstance(directions_raw, dict) else {}
    panel = Path(config.data.path or "")
    alpha_names = np.load(panel / "alpha_names.npy", allow_pickle=False).astype(str).tolist()
    result = {
        "status": "PASS",
        "factor_count": len(names),
        "factor_list_path": str(list_path),
        "factor_directions_path": str(direction_path),
        "factor_list_hash": _sha(list_path),
        "factor_directions_hash": _sha(direction_path),
        "same_fixed_names_as_panel": names == alpha_names,
        "direction_count": len(directions),
        "factor_reselection_invoked": False,
        "corr_selection_invoked": False,
        "all_folds_use_same_factor_list": True,
        "all_folds_use_same_directions": True,
    }
    result["status"] = "PASS" if result["factor_count"] == 64 and result["same_fixed_names_as_panel"] and result["direction_count"] == 64 else "FAIL"
    _json(output / "data_preflight" / "fixed_factor_audit.json", result)
    return result


def _make_fold_config(base: ExperimentConfig, fold: int, output: Path) -> ExperimentConfig:
    spec = dict(FOLDS[fold])
    metadata = _load_metadata(base)
    if fold == 4:
        spec["oos_end"] = str(metadata["latest_legal_execution_date"])
    config = copy.deepcopy(base)
    config.data.train_start = spec["train_start"]
    config.data.train_end = spec["train_end"]
    config.data.validation_start = spec["validation_start"]
    config.data.validation_end = spec["validation_end"]
    config.data.test_start = spec["oos_start"]
    config.data.test_end = spec["oos_end"]
    config.data.daily_validation_only = False
    config.data.purge_label_overlap = True
    config.data.label_exit_offsets = [2]
    config.logging.output_dir = str(output / f"fold_{fold}")
    config.validate()
    return config


def _fold_dates(config: ExperimentConfig, loaders: Any) -> dict[str, Any]:
    metadata = dict(loaders.dataset.split_metadata)
    used = metadata.get("used_date_ranges", {})
    raw = metadata.get("raw_counts", {})
    purged = metadata.get("purged_counts", {})
    return {
        "requested": {
            "train": [config.data.train_start, config.data.train_end],
            "validation": [config.data.validation_start, config.data.validation_end],
            "oos": [config.data.test_start, config.data.test_end],
        },
        "used": used,
        "counts": metadata.get("used_counts", {}),
        "raw_counts": raw,
        "purged_counts": purged,
        "purge_rule": metadata.get("purge_label_overlap"),
        "label_exit_offsets": metadata.get("label_exit_offsets"),
    }


def _assert_fold_isolation(loaders: Any) -> dict[str, Any]:
    split = loaders.dataset.split_metadata
    used = split.get("used_date_ranges", {})
    date_sets = {
        "train": {_canonical([loaders.dataset.dates[index]])[0] for index in loaders.train.dataset.indices},
        "validation": {_canonical([loaders.dataset.dates[index]])[0] for index in loaders.validation.dataset.indices},
        "oos": {_canonical([loaders.dataset.dates[index]])[0] for index in loaders.test.dataset.indices},
    }
    passed = not (date_sets["train"] & date_sets["validation"] or date_sets["train"] & date_sets["oos"] or date_sets["validation"] & date_sets["oos"])
    if not passed:
        raise RuntimeError("train/validation/OOS decision-date sets overlap")
    if not date_sets["oos"]:
        raise RuntimeError("OOS split has no legal decision dates")
    return {
        "status": "PASS",
        "decision_date_overlap": False,
        "oos_used_for_training": False,
        "oos_used_for_early_stopping": False,
        "used_ranges": used,
        "counts": {name: len(values) for name, values in date_sets.items()},
    }


def _write_config(path: Path, config: ExperimentConfig, extra: dict[str, Any] | None = None) -> None:
    value = config.to_dict()
    if extra:
        value.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _stage_a(config: ExperimentConfig, loaders: Any, destination: Path, device: torch.device) -> dict[str, Any]:
    """Formal A score pretraining: trainer score objective and A's Top20 evaluator."""
    destination.mkdir(parents=True, exist_ok=False)
    stage = copy.deepcopy(config)
    stage.loss.lambda_score = 1.0
    stage.loss.lambda_portfolio = 0.0
    stage.model.portfolio.allocation_mode = "topk_equal_weight"
    stage.model.portfolio.topk_ratio = 0.20
    stage.training.validation_target = "validation_daily_sharpe_7_5bps"
    _write_config(destination / "resolved_config.yaml", stage, {"stage": "A_score_pretraining", "selection_metric": "validation_daily_sharpe_7_5bps"})
    model = E2EAIModel(stage.model, stage.ablation).to(device)
    trainer = E2EAITrainer(model, stage, create_optimizer=True)
    try:
        fit = trainer.fit(loaders.train, loaders.validation, epochs=stage.training.epochs, output_dir=destination)
        trainer.load_checkpoint(destination / "best.pt", load_optimizer=False, restore_rng=False)
        metrics = trainer.evaluate(loaders.validation)
    finally:
        trainer.close()
    history = pd.DataFrame(fit.history)
    history.to_csv(destination / "training_history.csv", index=False, encoding="utf-8-sig")
    result = {
        "stage": "A",
        "best_epoch": int(fit.best_epoch),
        "best_metric": float(fit.best_metric) if fit.best_metric is not None else None,
        "epochs_completed": int(len(fit.history)),
        "optimizer_steps_total": int(len(loaders.train) * len(fit.history)),
        "early_stopped": bool(fit.stopped_early),
        "early_stop_reason": "validation_7p5bp_sharpe_patience" if fit.stopped_early else "max_epochs",
        "best_checkpoint_sha256": _sha(destination / "best.pt"),
        "validation_metrics": metrics,
    }
    _json(destination / "stage_metrics.json", result)
    return result


def _save_checkpoint(path: Path, model: E2EAIModel, optimizer: torch.optim.Optimizer, step: int, best_step: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_step": int(step),
            "best_optimizer_step": int(best_step),
            "best_validation_metric": float(best_metric),
        },
        path,
    )


def _load_state(model: E2EAIModel, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])


def _evaluation_files(evaluation: dict[str, Any], destination: Path, prefix: str) -> None:
    evaluation["score_frame"].to_parquet(destination / f"{prefix}_scores.parquet", index=False)
    evaluation["weight_frame"].to_parquet(destination / f"{prefix}_weights.parquet", index=False)
    evaluation["diagnostics"].to_csv(destination / f"{prefix}_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    evaluation["daily_returns"].to_csv(destination / f"{prefix}_daily_returns.csv", index=False, encoding="utf-8-sig")


def _stage_c(config: ExperimentConfig, source: Any, loaders: Any, destination: Path, a_checkpoint: Path, device: torch.device) -> dict[str, Any]:
    """Formal C gross-return fine tuning from the fold's own A checkpoint."""
    destination.mkdir(parents=True, exist_ok=False)
    stage = copy.deepcopy(config)
    stage.model.portfolio.allocation_mode = "long_only_softmax"
    stage.loss.lambda_score = 1.0
    stage.loss.lambda_portfolio = LAMBDA_PORTFOLIO
    stage.training.validation_target = "validation_daily_sharpe"
    _write_config(destination / "resolved_config.yaml", stage, {"stage": "C_gross_e2e", "lambda_portfolio": LAMBDA_PORTFOLIO, "allocator": "daily zscore then masked softmax, T=1.0", "selection_metric": "validation_daily_sharpe_0bps"})
    model = E2EAIModel(stage.model, stage.ablation).to(device)
    _load_state(model, a_checkpoint, device)
    shutil.copy2(a_checkpoint, destination / "initial_from_a.pt")
    optimizer = torch.optim.AdamW(_mlp_parameters(model), lr=stage.global_optimizer.lr, weight_decay=stage.global_optimizer.weight_decay)
    initial_eval = _evaluate(model, loaders.validation, source, device, stage)
    best_metric = float(initial_eval["metrics"]["validation_daily_sharpe_cost_0p0bps"])
    best_step = 0; step = 0; no_improve = 0; stopped = False
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    _save_checkpoint(destination / "checkpoints" / "best.pt", model, optimizer, 0, 0, best_metric)
    history: list[dict[str, Any]] = []
    steps_per_epoch = max(1, len(loaders.train))
    for epoch in range(stage.training.epochs):
        for raw in loaders.train:
            batch = _move_batch(dict(raw), device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            scores = _raw_scores(model, batch)
            score_loss, _ = _score_loss(scores, batch)
            gross_loss, _, _, _ = _gross_loss(scores, batch)
            loss = score_loss + LAMBDA_PORTFOLIO * gross_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite C loss")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(_mlp_parameters(model), stage.training.gradient_clip_norm, error_if_nonfinite=True))
            optimizer.step(); step += 1
            if step % 10:
                continue
            validation = _evaluate(model, loaders.validation, source, device, stage)
            metric = float(validation["metrics"]["validation_daily_sharpe_cost_0p0bps"])
            improved = metric > best_metric + 1e-12
            no_improve = 0 if improved else no_improve + 1
            if improved:
                best_metric = metric; best_step = step
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                _save_checkpoint(destination / "checkpoints" / "best.pt", model, optimizer, step, best_step, best_metric)
            history.append({"epoch": epoch + 1, "optimizer_step": step, "epoch_fraction": step / steps_per_epoch, "train_score_loss": float(score_loss.detach().cpu()), "train_gross_loss": float(gross_loss.detach().cpu()), "train_total_loss": float(loss.detach().cpu()), "gradient_norm": gradient, "validation_sharpe_0bps": metric, "validation_sharpe_7_5bps": validation["metrics"]["validation_daily_sharpe_cost_7p5bps"], "validation_rankic": validation["metrics"]["validation_execution_rankic"], "improved": improved, "no_improve": no_improve})
            print(f"[WF C] step={step} epoch={step / steps_per_epoch:.3f} val_sharpe_0bp={metric:.6f} improved={improved} no_improve={no_improve}", flush=True)
            if no_improve >= stage.training.early_stopping_patience:
                stopped = True; break
        if stopped:
            break
    _save_checkpoint(destination / "checkpoints" / "last.pt", model, optimizer, step, best_step, best_metric)
    model.load_state_dict(best_state)
    final = _evaluate(model, loaders.validation, source, device, stage)
    _evaluation_files(final, destination, "validation")
    pd.DataFrame(history).to_csv(destination / "training_steps.csv", index=False, encoding="utf-8-sig")
    metrics = {**final["metrics"], "stage": "C", "best_optimizer_step": best_step, "best_epoch_fraction": best_step / steps_per_epoch, "optimizer_steps_total": step, "early_stopped": stopped, "early_stop_reason": "validation_0bp_sharpe_patience" if stopped else "max_epochs", "lambda_portfolio": LAMBDA_PORTFOLIO, "initial_checkpoint_sha256": _sha(a_checkpoint), "best_checkpoint_sha256": _sha(destination / "checkpoints" / "best.pt")}
    _json(destination / "stage_metrics.json", metrics)
    return metrics


def _stage_d_stability(config: ExperimentConfig, source: Any, loaders: Any, destination: Path, c_checkpoint: Path, device: torch.device) -> dict[str, Any]:
    """D-Stability: overlap chronological net-return batches and fixed 7.5bp cost."""
    destination.mkdir(parents=True, exist_ok=False)
    stage = copy.deepcopy(config)
    stage.global_optimizer.lr = STABILITY_LR
    stage.model.portfolio.allocation_mode = "long_only_softmax"
    stage.loss.lambda_score = 1.0
    stage.loss.lambda_portfolio = LAMBDA_PORTFOLIO
    stage.training.validation_target = "validation_daily_sharpe_7_5bps"
    _write_config(destination / "resolved_config.yaml", stage, {"stage": "D_stability_net_e2e", "learning_rate": STABILITY_LR, "lambda_portfolio": LAMBDA_PORTFOLIO, "transaction_cost": "2 * turnover * 0.00075", "batch_logic": "chronological overlap"})
    train_batches = [dict(batch) for batch in loaders.train]
    specs = _batch_specs(train_batches)
    if not specs:
        raise RuntimeError("D-Stability received no chronological train batches")
    model = E2EAIModel(stage.model, stage.ablation).to(device)
    _load_state(model, c_checkpoint, device)
    shutil.copy2(c_checkpoint, destination / "initial_from_c.pt")
    optimizer = torch.optim.AdamW(_mlp_parameters(model), lr=STABILITY_LR, weight_decay=stage.global_optimizer.weight_decay)
    initial_eval = _evaluate(model, loaders.validation, source, device, stage)
    best_metric = float(initial_eval["metrics"]["validation_daily_sharpe_cost_7p5bps"])
    best_step = 0; step = 0; no_improve = 0; stopped = False
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    _save_checkpoint(destination / "checkpoints" / "best.pt", model, optimizer, 0, 0, best_metric)
    history: list[dict[str, Any]] = []
    for epoch in range(stage.training.epochs):
        for sequence, loss_start, _ in specs:
            batch = _collate_rows(sequence, device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            parts = _net_components(model, batch, loss_start)
            loss = parts["score_loss"] + LAMBDA_PORTFOLIO * parts["net_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite D-Stability loss")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(_mlp_parameters(model), stage.training.gradient_clip_norm, error_if_nonfinite=True))
            optimizer.step(); step += 1
            if step % 10:
                continue
            validation = _evaluate(model, loaders.validation, source, device, stage)
            metric = float(validation["metrics"]["validation_daily_sharpe_cost_7p5bps"])
            improved = metric > best_metric + 1e-12
            no_improve = 0 if improved else no_improve + 1
            if improved:
                best_metric = metric; best_step = step
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                _save_checkpoint(destination / "checkpoints" / "best.pt", model, optimizer, step, best_step, best_metric)
            history.append({"epoch": epoch + 1, "optimizer_step": step, "epoch_fraction": step / len(specs), "learning_rate": STABILITY_LR, "train_score_loss": float(parts["score_loss"].detach().cpu()), "train_net_loss": float(parts["net_loss"].detach().cpu()), "train_total_loss": float(loss.detach().cpu()), "train_turnover": float(parts["accounting"]["turnover"][loss_start:].mean().detach().cpu()), "train_cost_7_5bps": float(parts["accounting"]["cost_7_5bps"][loss_start:].mean().detach().cpu()), "gradient_norm": gradient, "validation_sharpe_7_5bps": metric, "validation_rankic": validation["metrics"]["validation_execution_rankic"], "improved": improved, "no_improve": no_improve})
            print(f"[WF D-Stability] step={step} epoch={step / len(specs):.3f} val_sharpe_7.5bp={metric:.6f} improved={improved} no_improve={no_improve}", flush=True)
            if no_improve >= stage.training.early_stopping_patience:
                stopped = True; break
        if stopped:
            break
    _save_checkpoint(destination / "checkpoints" / "last.pt", model, optimizer, step, best_step, best_metric)
    model.load_state_dict(best_state)
    final = _evaluate(model, loaders.validation, source, device, stage)
    _evaluation_files(final, destination, "validation")
    pd.DataFrame(history).to_csv(destination / "training_steps.csv", index=False, encoding="utf-8-sig")
    metrics = {**final["metrics"], "stage": "D-Stability", "best_optimizer_step": best_step, "best_epoch_fraction": best_step / len(specs), "optimizer_steps_total": step, "early_stopped": stopped, "early_stop_reason": "validation_7p5bp_sharpe_patience" if stopped else "max_epochs", "learning_rate": STABILITY_LR, "lambda_portfolio": LAMBDA_PORTFOLIO, "initial_checkpoint_sha256": _sha(c_checkpoint), "best_checkpoint_sha256": _sha(destination / "checkpoints" / "best.pt")}
    _json(destination / "stage_metrics.json", metrics)
    return metrics


def _scores_to_oos_outputs(evaluation: dict[str, Any], source: Any, destination: Path, *, diagnostic_only: bool) -> dict[str, Any]:
    score = evaluation["score_frame"].copy()
    score["date"] = _canonical_dates(score["date"])
    weights, diagnostics, targets = _build_weights(score)
    weights = weights.rename(columns={"weight": "target_weight", "execution_1d_return": "execution_return"})
    required = ["date", "asset", "raw_score", "normalized_score", "target_weight", "valid_asset", "execution_return"]
    weights = weights[required]
    dates = sorted(weights["date"].unique().tolist())
    execution = []
    for _, group in weights.groupby("date", sort=True):
        execution.append({str(row.asset): float(row.execution_return) for row in group.itertuples(index=False) if np.isfinite(row.execution_return)})
    benchmark = _benchmark_for_dates(source, dates)
    targets = [{str(row.asset): float(row.target_weight) for row in group.itertuples(index=False) if bool(row.valid_asset)} for _, group in weights.groupby("date", sort=True)]
    performance, daily = _performance(targets, execution, benchmark, [0.0, COST_BPS], annual_trading_days=252, holding_threshold=1e-8)
    daily.insert(0, "date", dates)
    daily["cost_7_5bps"] = 2.0 * daily["one_way_turnover"] * COST_BPS / 10000.0
    daily["net_return_7_5bps"] = daily["net_return_7p5bps"]
    weights.to_parquet(destination / "oos_target_weights.parquet", index=False)
    score.to_parquet(destination / "oos_scores.parquet", index=False)
    daily.to_csv(destination / "oos_daily_returns_isolated.csv", index=False, encoding="utf-8-sig")
    metrics = {"status": "DIAGNOSTIC_ONLY" if diagnostic_only else "FORMAL", "dates": [dates[0], dates[-1]], "date_count": len(dates), **performance}
    _json(destination / "oos_metrics_isolated.json", metrics)
    return metrics


def _full_rankic(model: E2EAIModel, loader: Any, device: torch.device) -> float:
    scores = _collect_scores(model, loader, device)
    values: list[float] = []
    for _, group in scores.groupby("date", sort=True):
        x = pd.to_numeric(group["raw_score"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group["execution_1d_return"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) >= 3:
            values.append(_rank_corr(x[valid], y[valid]))
    return float(np.mean(values)) if values else 0.0


def run_fold(base: ExperimentConfig, output: Path, fold: int) -> Path:
    if not _preflight_path(output).is_file() or json.loads(_preflight_path(output).read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("walk-forward data preflight has not passed; refusing to train")
    destination = _root(output) / f"fold_{fold}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing fold output: {destination}")
    config = _make_fold_config(base, fold, output)
    seed_everything(config.seed)
    source, _ = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    isolation = _assert_fold_isolation(loaders)
    device = resolve_device(config.device)
    destination.mkdir(parents=True, exist_ok=False)
    _write_config(destination / "resolved_config.yaml", config, {"fold": fold, "fixed_lambda_portfolio": LAMBDA_PORTFOLIO, "temperature": TEMPERATURE, "no_factor_reselection": True})
    _json(destination / "split_isolation.json", isolation)
    a = _stage_a(config, loaders, destination / "stage_a", device)
    c = _stage_c(config, source, loaders, destination / "stage_c", destination / "stage_a" / "best.pt", device)
    d = _stage_d_stability(config, source, loaders, destination / "stage_d_stability", destination / "stage_c" / "checkpoints" / "best.pt", device)
    model = E2EAIModel(config.model, config.ablation).to(device)
    _load_state(model, destination / "stage_d_stability" / "checkpoints" / "best.pt", device)
    validation = _evaluate(model, loaders.validation, source, device, config)
    validation["weight_frame"].to_parquet(destination / "validation_weights.parquet", index=False)
    validation["daily_returns"].to_csv(destination / "validation_daily_returns.csv", index=False, encoding="utf-8-sig")
    validation_metrics = {**validation["metrics"], "selection_checkpoint": "stage_d_stability/checkpoints/best.pt", "selection_metric": "validation_daily_sharpe_7_5bps", "formal_oos_note": "OOS isolated values are diagnostic only; annual official values come from the continuous stitcher."}
    _json(destination / "validation_metrics.json", validation_metrics)
    oos = _evaluate(model, loaders.test, source, device, config)
    oos_metrics = _scores_to_oos_outputs(oos, source, destination, diagnostic_only=True)
    audit = {"evaluator": "daily cross-sectional Spearman RankIC", "full_train_rankic": _full_rankic(model, loaders.train, device), "full_validation_rankic": _full_rankic(model, loaders.validation, device)}
    audit["true_rankic_gap"] = audit["full_train_rankic"] - audit["full_validation_rankic"]
    _json(destination / "full_period_rankic_audit.json", audit)
    shutil.copy2(destination / "stage_d_stability" / "checkpoints" / "best.pt", destination / "best_checkpoint.pt")
    fixed_audit = json.loads((output / "data_preflight" / "fixed_factor_audit.json").read_text(encoding="utf-8"))
    manifest = {"status": "PASS", "fold": fold, "created_at_utc": datetime.now(UTC).isoformat(), "device": str(device), "fold_dates": _fold_dates(config, loaders), "split_isolation": isolation, "fixed_factor_list_sha256": fixed_audit["factor_list_hash"], "fixed_factor_directions_sha256": fixed_audit["factor_directions_hash"], "stage_a": {key: a.get(key) for key in ("best_epoch", "best_metric", "epochs_completed", "optimizer_steps_total", "early_stopped", "best_checkpoint_sha256")}, "stage_c": {key: c.get(key) for key in ("best_optimizer_step", "optimizer_steps_total", "early_stopped", "best_checkpoint_sha256")}, "stage_d_stability": {key: d.get(key) for key in ("best_optimizer_step", "optimizer_steps_total", "early_stopped", "best_checkpoint_sha256", "learning_rate")}, "oos_diagnostic_only": True, "no_factor_reselection": True, "no_temperature_tuning": True, "no_new_portfolio_constraint": True}
    _json(destination / "fold_manifest.json", manifest)
    print(f"[WF] fold={fold} PASS output={destination}", flush=True)
    return destination


def _synthetic_boundary_test() -> dict[str, Any]:
    targets = [{"a": 0.9, "b": 0.1}, {"a": 0.8, "b": 0.2}, {"a": 0.05, "b": 0.95}, {"a": 0.1, "b": 0.9}]
    returns = [{"a": 0.03, "b": -0.02}, {"a": -0.01, "b": 0.04}, {"a": 0.02, "b": -0.01}, {"a": 0.01, "b": 0.0}]
    drift, _, turnover, _, _ = numpy_drifted_rows(targets, returns)
    expected = 0.5 * sum(abs(targets[2].get(asset, 0.0) - drift[2].get(asset, 0.0)) for asset in set(targets[2]) | set(drift[2]))
    _, _, separate_turnover, _, _ = numpy_drifted_rows(targets[2:], returns[2:])
    passed = bool(turnover[2] > 0 and abs(turnover[2] - expected) < 1e-12 and separate_turnover[0] == 0.0)
    return {"status": "PASS" if passed else "FAIL", "switch_turnover": float(turnover[2]), "expected_turnover": float(expected), "separate_year_incorrect_first_turnover": float(separate_turnover[0]), "definition": "0.5 * L1(new target - drifted previous target)"}


def run_preflight(base: ExperimentConfig, output: Path) -> Path:
    root = _root(output); destination = root / "data_preflight"; destination.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata(base)
    audit = _fixed_factor_audit(base, root)
    panel = Path(base.data.path or "")
    identity_path = panel / "overlap_identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(f"panel overlap identity is missing: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    _json(destination / "panel_overlap_identity.json", identity)
    availability = {"status": "PASS" if metadata.get("execution_return", {}).get("latest_valid_decision_date") else "FAIL", **metadata.get("execution_return", {}), "oos_2026_start_date": "2026-01-01"}
    dates = pd.to_datetime(np.load(panel / "dates.npy", allow_pickle=False))
    availability["oos_2026_number_of_decision_dates"] = int(((dates >= pd.Timestamp("2026-01-01")) & (dates <= pd.Timestamp(availability["latest_valid_decision_date"]))).sum())
    _json(destination / "oos_2026_availability.json", availability)
    folds: dict[str, Any] = {}
    for fold in FOLDS:
        config = _make_fold_config(base, fold, root)
        source, _ = load_configured_data(config)
        loaders = build_dataloaders(config, source)
        folds[str(fold)] = {"dates": _fold_dates(config, loaders), "isolation": _assert_fold_isolation(loaders)}
    _json(destination / "fold_date_audit.json", folds)
    # Accounting parity uses exactly the accepted drift-aware reference, and the
    # synthetic test proves a year switch cannot be zeroed for free.
    targets = [{"a": 0.6, "b": 0.4}, {"a": 0.4, "b": 0.6}, {"a": 0.7, "b": 0.3}]
    returns = [{"a": 0.01, "b": -0.01}, {"a": -0.02, "b": 0.03}, {"a": 0.01, "b": 0.02}]
    path = _simulate_daily_weight_path(targets, returns, cost_bps=COST_BPS, holding_threshold=1e-8)
    _, gross, turnover, cost, net = numpy_drifted_rows(targets, returns)
    accounting = {"status": "PASS" if np.max(np.abs(path["gross_returns"] - gross)) < 1e-12 and np.max(np.abs(path["turnovers"] - turnover)) < 1e-12 and np.max(np.abs(path["net_returns"] - net)) < 1e-12 else "FAIL", "max_gross_diff": float(np.max(np.abs(path["gross_returns"] - gross))), "max_turnover_diff": float(np.max(np.abs(path["turnovers"] - turnover))), "max_cost_diff": float(np.max(np.abs((path["gross_returns"] - path["net_returns"]) - cost)))}
    boundary = _synthetic_boundary_test()
    _json(destination / "transaction_cost_accounting_test.json", accounting)
    _json(destination / "synthetic_stitch_boundary_test.json", boundary)
    checks = {"fixed64": audit["status"], "overlap_identity": identity.get("status"), "return_timing": "PASS" if metadata.get("execution_return", {}).get("definition") == "adjusted VWAP[t+2] / adjusted VWAP[t+1] - 1" else "FAIL", "benchmark_timing": "PASS" if metadata.get("execution_return", {}).get("benchmark_weights_date") == "t decision-date CSI500 weights" else "FAIL", "fold_dates": "PASS" if all(value["isolation"]["status"] == "PASS" for value in folds.values()) else "FAIL", "oos_2026": availability["status"], "transaction_cost": accounting["status"], "synthetic_stitch_boundary": boundary["status"]}
    result = {"status": "PASS" if all(value == "PASS" for value in checks.values()) else "FAIL", "created_at_utc": datetime.now(UTC).isoformat(), "checks": checks, "no_factor_reselection": True, "no_corr_selection": True, "fixed_factor_hashes": {"factor_list": audit["factor_list_hash"], "directions": audit["factor_directions_hash"]}}
    _json(destination / "preflight.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"walk-forward preflight failed: {result}")
    return destination


def _available_gpus() -> int:
    return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0


def run_all(config_path: Path, output: Path, parallel: bool) -> None:
    if not _preflight_path(output).is_file():
        raise RuntimeError("run --preflight before --all")
    gpu_count = _available_gpus()
    workers = min(5, gpu_count) if parallel and gpu_count > 1 else 1
    mapping = {str(fold): (fold % gpu_count if gpu_count else None) for fold in FOLDS}
    _json(_root(output) / "fold_to_gpu_mapping.json", {"parallel_requested": parallel, "available_gpu_count": gpu_count, "worker_count": workers, "fold_to_gpu": mapping})
    commands = []
    for fold in FOLDS:
        env = os.environ.copy()
        if gpu_count:
            env["CUDA_VISIBLE_DEVICES"] = str(mapping[str(fold)])
        commands.append((fold, env))
    if workers == 1:
        for fold, env in commands:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--output", str(output), "--fold", str(fold)], check=True, env=env)
        return
    active: list[tuple[int, subprocess.Popen[Any]]] = []
    pending = list(commands)
    while pending or active:
        while pending and len(active) < workers:
            fold, env = pending.pop(0)
            active.append((fold, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--output", str(output), "--fold", str(fold)], env=env)))
        for fold, process in list(active):
            code = process.poll()
            if code is not None:
                active.remove((fold, process))
                if code:
                    raise RuntimeError(f"fold {fold} failed with exit code {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fixed64_annual_walk_forward_baseline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/cloud/E2EAI/output/fixed64_annual_walk_forward_baseline"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--fold", type=int, choices=sorted(FOLDS))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    if args.preflight:
        print(run_preflight(base, args.output))
    elif args.fold is not None:
        print(run_fold(base, args.output, args.fold))
    else:
        run_all(args.config.resolve(), args.output, args.parallel)


if __name__ == "__main__":
    main()
