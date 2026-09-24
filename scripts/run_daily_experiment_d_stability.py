"""Run the isolated Experiment D-Stability (AdamW learning rate 1e-4).

This diagnostic starts from the frozen Experiment C best checkpoint and keeps
Experiment D's architecture, daily accounting, loss, chronology and
validation protocol unchanged.  It writes a new output directory only; the
formal Experiment D artifacts are read-only inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.models.e2eai import E2EAIModel
from e2eai.utils.reproducibility import resolve_device, seed_everything
from e2eai.workflows import load_configured_data, write_metrics
from scripts.run_daily_experiment_b import _canonical_dates, _sha256
from scripts.run_daily_experiment_c import _collect_scores, _rank_corr
from scripts.run_daily_experiment_d import (
    COST_BPS,
    LAMBDA_PORTFOLIO,
    _batch_specs,
    _collate_rows,
    _evaluate,
    _net_components,
    _preflight_accounting,
    _step0_gradient_decomposition,
    _stability_diagnostics,
    _daily_with_costs,
    _max_abs,
    _mlp_parameters,
)

STABILITY_LR = 1e-4
REFERENCE_D_LR = 1e-3
TOLERANCE = 1e-6


def _state_vector(model: E2EAIModel) -> np.ndarray:
    values = [parameter.detach().cpu().numpy().astype(np.float64).reshape(-1) for parameter in _mlp_parameters(model)]
    return np.concatenate(values) if values else np.zeros(0, dtype=np.float64)


def _relative_distance(model: E2EAIModel, initial: np.ndarray) -> tuple[float, float]:
    distance = float(np.linalg.norm(_state_vector(model) - initial))
    denominator = max(float(np.linalg.norm(initial)), 1e-12)
    return distance, distance / denominator


def _gradient_norm(model: E2EAIModel) -> float:
    values = [p.grad.detach().cpu().numpy().astype(np.float64).reshape(-1) for p in _mlp_parameters(model) if p.grad is not None]
    if not values:
        return 0.0
    return float(np.linalg.norm(np.concatenate(values)))


def _save_checkpoint(path: Path, model: E2EAIModel, optimizer: torch.optim.Optimizer, step: int, best_step: int, best_metric: float | None) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": int(step),
            "best_epoch": int(best_step),
            "best_validation_metric": best_metric,
            "training_experiment": "experiment_d_stability",
            "learning_rate": STABILITY_LR,
        },
        path,
    )


def _artifact_identity(result: dict[str, Any], reference_output: Path) -> dict[str, Any]:
    """Compare a fresh step-0 evaluation to a frozen C/D validation artifact."""
    reference_weights = pd.read_parquet(reference_output / "validation_weights.parquet")
    right = result["weight_frame"].copy()
    left = reference_weights.rename(columns={"asset_id": "asset"}).copy()
    cols = ["date", "asset", "raw_score", "normalized_score", "weight", "valid_asset"]
    left = left[cols]; right = right[cols]
    for frame in (left, right):
        frame["date"] = _canonical_dates(frame["date"])
        frame["asset"] = frame["asset"].astype(str)
    left = left.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    right = right.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    same_axis = left[["date", "asset"]].astype(str).equals(right[["date", "asset"]].astype(str))
    diffs: dict[str, float | None] = {}
    if len(left) == len(right):
        for column in ["raw_score", "normalized_score", "weight"]:
            diffs[f"{column}_max_abs_diff"] = _max_abs(left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float))
    else:
        for column in ["raw_score", "normalized_score", "weight"]:
            diffs[f"{column}_max_abs_diff"] = None
    reference_daily = pd.read_csv(reference_output / "validation_daily_returns.csv")
    current_daily = result["daily_returns"].copy()
    reference_daily["date"] = _canonical_dates(reference_daily["date"])
    current_daily["date"] = _canonical_dates(current_daily["date"])
    merged = reference_daily.merge(current_daily, on="date", suffixes=("_ref", "_new"), validate="one_to_one")
    for column in ["gross_return", "one_way_turnover", "net_return_7p5bps"]:
        diffs[f"{column}_max_abs_diff"] = _max_abs(merged[f"{column}_ref"].to_numpy(), merged[f"{column}_new"].to_numpy())
    diffs["cost_7p5bps_max_abs_diff"] = 2.0 * COST_BPS / 10000.0 * float(diffs["one_way_turnover_max_abs_diff"])
    passed = bool(same_axis and all(value is not None and value <= TOLERANCE for value in diffs.values()))
    return {
        "status": "PASS" if passed else "FAIL",
        "reference_output": str(reference_output),
        "same_date_asset_axis": same_axis,
        "tolerance": TOLERANCE,
        **diffs,
    }


def _d_step0_metric_identity(result: dict[str, Any], d_output: Path) -> dict[str, Any]:
    """Compare to D's recorded step-0 validation row.

    Experiment D's persisted validation weights are its selected best model,
    not a step-0 snapshot.  Its training_steps.csv does retain the complete
    step-0 validation metrics, so those are the valid D-side identity check.
    """
    history = pd.read_csv(d_output / "training_steps.csv")
    row = history.loc[history["optimizer_step"] == 0]
    if row.empty:
        return {"status": "FAIL", "reason": "D training_steps.csv has no step-0 row"}
    row = row.iloc[0]
    mapping = {
        "validation_rankic": "validation_execution_rankic",
        "validation_annual_return_0bps": "validation_daily_return_cost_0p0bps",
        "validation_sharpe_0bps": "validation_daily_sharpe_cost_0p0bps",
        "validation_annual_return_7_5bps": "validation_daily_return_cost_7p5bps",
        "validation_sharpe_7_5bps": "validation_daily_sharpe_cost_7p5bps",
        "validation_mdd_7_5bps": "validation_daily_mdd_cost_7p5bps",
        "validation_turnover": "validation_daily_turnover",
        "validation_effective_n": "validation_effective_n",
    }
    diffs = {f"{column}_max_abs_diff": abs(float(result["metrics"][metric]) - float(row[column])) for column, metric in mapping.items()}
    return {"status": "PASS" if all(value <= TOLERANCE for value in diffs.values()) else "FAIL", "reference_output": str(d_output), "source": "training_steps.csv optimizer_step=0", "tolerance": TOLERANCE, "raw_score_artifact_available": False, **diffs}


def _full_period_rankic(score_frame: pd.DataFrame) -> float:
    values: list[float] = []
    for _, group in score_frame.groupby("date", sort=True):
        scores = pd.to_numeric(group["raw_score"], errors="coerce").to_numpy(dtype=np.float64)
        returns = pd.to_numeric(group["execution_1d_return"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(scores) & np.isfinite(returns)
        if int(valid.sum()) >= 3:
            values.append(_rank_corr(scores[valid], returns[valid]))
    return float(np.mean(values)) if values else 0.0


def _audit_checkpoint(model: E2EAIModel, checkpoint: Path, train_loader: Any, validation_loader: Any, device: torch.device, initial: np.ndarray) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    distance, relative = _relative_distance(model, initial)
    train_scores = _collect_scores(model, train_loader, device)
    validation_scores = _collect_scores(model, validation_loader, device)
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "full_train_rankic": _full_period_rankic(train_scores),
        "full_validation_rankic": _full_period_rankic(validation_scores),
        "true_rankic_gap": _full_period_rankic(train_scores) - _full_period_rankic(validation_scores),
        "parameter_l2_distance": distance,
        "relative_parameter_l2_distance": relative,
        "train_date_start": str(train_scores["date"].min())[:10],
        "train_date_end": str(train_scores["date"].max())[:10],
        "validation_date_start": str(validation_scores["date"].min())[:10],
        "validation_date_end": str(validation_scores["date"].max())[:10],
        "train_date_count": int(train_scores["date"].nunique()),
        "validation_date_count": int(validation_scores["date"].nunique()),
    }


def _near_best(history: pd.DataFrame, threshold_gap: float) -> dict[str, Any]:
    values = history[["optimizer_step", "validation_sharpe_7_5bps"]].dropna().sort_values("optimizer_step")
    if values.empty:
        return {"threshold_gap": threshold_gap, "threshold": None, "checkpoint_count": 0, "longest_consecutive_count": 0, "steps": []}
    best = float(values["validation_sharpe_7_5bps"].max())
    threshold = best - threshold_gap
    flags = (values["validation_sharpe_7_5bps"] >= threshold).to_numpy(dtype=bool)
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return {"threshold_gap": threshold_gap, "best_sharpe": best, "threshold": threshold, "checkpoint_count": int(flags.sum()), "longest_consecutive_count": int(longest), "steps": values.loc[flags, "optimizer_step"].astype(int).tolist()}


def _local_deterioration(history: pd.DataFrame) -> dict[str, Any]:
    rows = history.set_index("optimizer_step")
    best_row = history.loc[history["validation_sharpe_7_5bps"].idxmax()]
    best_step = int(best_row["optimizer_step"])
    result: dict[str, Any] = {"best_step": best_step, "best_sharpe": float(best_row["validation_sharpe_7_5bps"]), "offsets": {}}
    for offset in [10, 20, 50, 100]:
        target = best_step + offset
        if target not in rows.index:
            result["offsets"][f"plus_{offset}"] = {"status": "UNAVAILABLE"}
            continue
        row = rows.loc[target]
        result["offsets"][f"plus_{offset}"] = {
            "status": "AVAILABLE",
            "optimizer_step": target,
            "delta_validation_sharpe": float(row["validation_sharpe_7_5bps"] - best_row["validation_sharpe_7_5bps"]),
            "delta_validation_rankic": float(row["validation_rankic"] - best_row["validation_rankic"]),
            "delta_turnover": float(row["validation_turnover"] - best_row["validation_turnover"]),
        }
    return result


def _param_update_record(model: E2EAIModel, initial: np.ndarray, step: int, epoch_fraction: float, learning_rate: float, pre_clip: float, post_clip: float, before: np.ndarray) -> dict[str, Any]:
    now = _state_vector(model)
    update = float(np.linalg.norm(now - before))
    distance = float(np.linalg.norm(now - initial))
    denominator = max(float(np.linalg.norm(now)), 1e-12)
    return {
        "optimizer_step": step,
        "epoch_fraction": epoch_fraction,
        "learning_rate": learning_rate,
        "pre_clip_gradient_norm": pre_clip,
        "post_clip_gradient_norm": post_clip,
        "parameter_update_l2_norm": update,
        "relative_update_norm": update / denominator,
        "parameter_l2_distance": distance,
        "relative_parameter_l2_distance": distance / max(float(np.linalg.norm(initial)), 1e-12),
    }


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_comparison(output: Path, d_output: Path, stability_metrics: dict[str, Any], audits: pd.DataFrame, d_history: pd.DataFrame, stability_history: pd.DataFrame, d_initial: np.ndarray, d_model: E2EAIModel, stability_initial: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    d_metrics = _load_metrics(d_output / "validation_metrics.json")
    d_audit = audits[audits["experiment"] == "experiment_d"].set_index("checkpoint")
    s_audit = audits[audits["experiment"] == "experiment_d_stability"].set_index("checkpoint")
    d_best_step = int(d_metrics["best_optimizer_step"]); s_best_step = int(stability_metrics["best_optimizer_step"])
    d_best = d_history.loc[d_history["optimizer_step"] == d_best_step].iloc[0]
    s_best = stability_history.loc[stability_history["optimizer_step"] == s_best_step].iloc[0]
    rows = []
    def add(metric: str, d_value: Any, s_value: Any) -> None:
        rows.append({"metric": metric, "D": d_value, "D_stability": s_value, "Stability_minus_D": None if d_value is None or s_value is None else float(s_value) - float(d_value)})
    add("learning_rate", REFERENCE_D_LR, STABILITY_LR)
    add("best_optimizer_step", d_best_step, s_best_step)
    add("best_epoch_fraction", d_metrics.get("best_epoch_fraction"), stability_metrics.get("best_epoch_fraction"))
    add("total_optimizer_steps", d_metrics.get("optimizer_steps_total"), stability_metrics.get("optimizer_steps_total"))
    for name, d_col, s_col in [
        ("best_7.5bp_annual_return", "validation_annual_return_7_5bps", "validation_annual_return_7_5bps"),
        ("best_7.5bp_sharpe", "validation_sharpe_7_5bps", "validation_sharpe_7_5bps"),
        ("best_7.5bp_mdd", "validation_mdd_7_5bps", "validation_mdd_7_5bps"),
        ("best_0bp_sharpe", "validation_sharpe_0bps", "validation_sharpe_0bps"),
        ("best_turnover", "validation_turnover", "validation_turnover"),
        ("best_effective_n", "validation_effective_n", "validation_effective_n"),
        ("best_max_weight", "validation_max_weight", "validation_max_weight"),
        ("best_information_ratio", "validation_information_ratio_7_5bps", "validation_information_ratio_7_5bps"),
    ]:
        add(name, d_best.get(d_col), s_best.get(s_col))
    for metric, checkpoint in [("full_train_rankic", "best.pt"), ("full_validation_rankic", "best.pt"), ("true_rankic_gap", "best.pt")]:
        add(metric, float(d_audit.loc[checkpoint, metric]), float(s_audit.loc[checkpoint, metric]))
    add("best_relative_parameter_distance", float(d_audit.loc["best.pt", "relative_parameter_l2_distance"]), float(s_audit.loc["best.pt", "relative_parameter_l2_distance"]))
    d_near02 = _near_best(d_history, 0.02); s_near02 = _near_best(stability_history, 0.02)
    d_near05 = _near_best(d_history, 0.05); s_near05 = _near_best(stability_history, 0.05)
    add("near_best_minus_0.02_checkpoint_count", d_near02["checkpoint_count"], s_near02["checkpoint_count"])
    add("near_best_minus_0.02_longest_count", d_near02["longest_consecutive_count"], s_near02["longest_consecutive_count"])
    add("near_best_minus_0.05_checkpoint_count", d_near05["checkpoint_count"], s_near05["checkpoint_count"])
    add("near_best_minus_0.05_longest_count", d_near05["longest_consecutive_count"], s_near05["longest_consecutive_count"])
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "D_vs_D_stability.csv", index=False, encoding="utf-8-sig")
    lines = ["# Experiment D vs D-Stability", "", "Only the AdamW learning rate changes: 1e-3 in D versus 1e-4 in D-Stability.", "", "| Metric | D | D-Stability | Stability − D |", "|---|---:|---:|---:|"]
    lines.extend(f"| {row['metric']} | {row['D']} | {row['D_stability']} | {row['Stability_minus_D']} |" for _, row in comparison.iterrows())
    (output / "D_vs_D_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "D_best_region_minus_0.02": d_near02,
        "D_stability_best_region_minus_0.02": s_near02,
        "D_best_region_minus_0.05": d_near05,
        "D_stability_best_region_minus_0.05": s_near05,
        "D_local_deterioration": _local_deterioration(d_history),
        "D_stability_local_deterioration": _local_deterioration(stability_history),
        "best_step_moved_later": s_best_step > d_best_step,
        "best_sharpe_delta": float(s_best["validation_sharpe_7_5bps"] - d_best["validation_sharpe_7_5bps"]),
        "best_net_return_delta": float(s_best["validation_annual_return_7_5bps"] - d_best["validation_annual_return_7_5bps"]),
        "true_rankic_gap_delta": float(s_audit.loc["best.pt", "true_rankic_gap"] - d_audit.loc["best.pt", "true_rankic_gap"]),
        "mean_update_norm_D_unavailable": True,
        "mean_update_norm_D_stability": float(pd.read_csv(output / "parameter_distance.csv")["parameter_update_l2_norm"].dropna().mean()),
    }
    return comparison, summary


def _finalize_existing(config_path: Path, *, c_output: Path, d_output: Path, output: Path) -> Path:
    """Complete artifacts after a non-model post-processing interruption."""
    config = load_config(config_path)
    source, _ = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    device = resolve_device(config.device)
    audit = pd.read_csv(output / "full_period_metric_audit.csv")
    stability_history = pd.read_csv(output / "training_steps.csv")
    d_history = pd.read_csv(d_output / "training_steps.csv")
    reported_d_train = _load_metrics(d_output / "validation_metrics.json").get("train_rankic_at_best")
    audit_table = audit.to_csv(index=False)
    (output / "full_period_metric_audit.md").write_text("\n".join(["# Full-Period RankIC Audit", "", "All checkpoints use the same full-period Spearman RankIC evaluator over complete train and validation decision dates.", "", "```csv", audit_table.rstrip(), "```", "", f"Original Experiment D reported training-log RankIC: {reported_d_train}", "The reported value is a best-step mini-batch/rolling training-log metric, not the full-period evaluator metric."]) + "\n", encoding="utf-8")
    d_learning = d_history[["optimizer_step", "epoch_fraction", "validation_sharpe_7_5bps", "validation_annual_return_7_5bps", "validation_rankic", "validation_turnover", "validation_effective_n", "validation_information_ratio_7_5bps"]].copy()
    d_learning = d_learning.rename(columns={column: f"D_{column}" for column in d_learning.columns if column != "optimizer_step"})
    s_learning = stability_history[["optimizer_step", "epoch_fraction", "learning_rate", "validation_sharpe_7_5bps", "validation_annual_return_7_5bps", "validation_rankic", "validation_turnover", "validation_effective_n", "validation_information_ratio_7_5bps", "relative_parameter_l2_distance", "parameter_update_l2_norm"]].copy()
    s_learning = s_learning.rename(columns={column: f"D_stability_{column}" for column in s_learning.columns if column != "optimizer_step"})
    curve = d_learning.merge(s_learning, on="optimizer_step", how="outer").sort_values("optimizer_step")
    d_audit = audit[audit["experiment"] == "experiment_d"].set_index("checkpoint")
    d_metrics = _load_metrics(d_output / "validation_metrics.json")
    known_d_distances = {0: float(d_audit.loc["initial_from_c.pt", "relative_parameter_l2_distance"]), int(d_metrics["best_optimizer_step"]): float(d_audit.loc["best.pt", "relative_parameter_l2_distance"]), int(d_metrics["optimizer_steps_total"]): float(d_audit.loc["last.pt", "relative_parameter_l2_distance"])}
    curve["D_relative_parameter_l2_distance"] = curve["optimizer_step"].map(known_d_distances)
    curve.to_csv(output / "D_vs_D_stability_learning_curve.csv", index=False, encoding="utf-8-sig")
    stability_metrics = _load_metrics(output / "validation_metrics.json")
    _, summary = _build_comparison(output, d_output, stability_metrics, audit, d_history, stability_history, np.zeros(1), None, np.zeros(1))
    summary.update({"preflight_status": _load_metrics(output / "transaction_cost_preflight.json")["status"], "step0_identity_status": _load_metrics(output / "step0_identity.json")["status"], "reported_d_train_rankic": reported_d_train, "full_period_audit": audit.to_dict("records"), "only_training_variable_learning_rate": True, "d_learning_rate": REFERENCE_D_LR, "d_stability_learning_rate": STABILITY_LR, "weight_decay_same": True, "lambda_same": True, "temperature_same": True, "seed_same": True, "batch_logic_same": True, "no_2022": True, "experiment_e_run": False})
    write_metrics(output / "stability_summary.json", summary)
    manifest = {"experiment": "experiment_d_stability", "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_experiment_c": str(c_output), "source_experiment_d": str(d_output), "output_dir": str(output), "c_best_checkpoint_sha256": _sha256(c_output / "checkpoints" / "best.pt"), "d_initial_checkpoint_sha256": _sha256(d_output / "initial_from_c.pt"), "stability_initial_checkpoint_sha256": _sha256(output / "initial_from_c.pt"), "initial_hashes_match": _sha256(c_output / "checkpoints" / "best.pt") == _sha256(d_output / "initial_from_c.pt") == _sha256(output / "initial_from_c.pt"), "step0_identity_status": _load_metrics(output / "step0_identity.json")["status"], "preflight_status": _load_metrics(output / "transaction_cost_preflight.json")["status"], "learning_rate": STABILITY_LR, "reference_d_learning_rate": REFERENCE_D_LR, "weight_decay": config.global_optimizer.weight_decay, "lambda_portfolio": LAMBDA_PORTFOLIO, "temperature": 1.0, "transaction_cost_bps": COST_BPS, "extra_turnover_penalty": 0.0, "no_2022": True, "experiment_e_run": False, "optimizer_steps_total": int(stability_metrics["optimizer_steps_total"]), "best_optimizer_step": int(stability_metrics["best_optimizer_step"]), "early_stopped": bool(stability_metrics["early_stopped"]), "files": sorted({path.name for path in output.iterdir()} | {"run_manifest.json"})}
    write_metrics(output / "run_manifest.json", manifest)
    return output


def run(config_path: Path, *, c_output: Path, d_output: Path, output: Path) -> Path:
    c_output = c_output.resolve(); d_output = d_output.resolve(); output = output.resolve()
    c_checkpoint = c_output / "checkpoints" / "best.pt"
    if not c_checkpoint.is_file():
        raise FileNotFoundError(c_checkpoint)
    if not (d_output / "validation_metrics.json").is_file():
        raise FileNotFoundError(d_output / "validation_metrics.json")
    config = load_config(config_path)
    if config.model.horizons != [2] or config.model.execution_horizon != 2:
        raise ValueError("D-Stability requires h=2")
    if config.data.test_start is not None or config.data.test_end is not None:
        raise ValueError("D-Stability must not configure OOS/test")
    config.global_optimizer.lr = STABILITY_LR
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copy2(c_checkpoint, output / "initial_from_c.pt")
    seed_everything(config.seed)
    source, _ = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    device = resolve_device(config.device)
    train_batches = [dict(batch) for batch in loaders.train]
    specs = _batch_specs(train_batches)
    train_dates = [str(row["date"])[:10] for batch in train_batches for row in __import__("scripts.run_daily_experiment_d", fromlist=["_rows_from_batch"])._rows_from_batch(batch)]
    if any(date >= "2021-01-01" for date in train_dates):
        raise RuntimeError("D-Stability train loader includes validation/test dates")
    model = E2EAIModel(config.model, config.ablation).to(device)
    payload = torch.load(c_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    initial = _state_vector(model)
    step0 = _evaluate(model, loaders.validation, source, device, config)
    identity_c = _artifact_identity(step0, c_output)
    identity_d = _d_step0_metric_identity(step0, d_output)
    identity = {"status": "PASS" if identity_c["status"] == "PASS" and identity_d["status"] == "PASS" else "FAIL", "c_best_checkpoint_sha256": _sha256(c_checkpoint), "d_initial_checkpoint_sha256": _sha256(d_output / "initial_from_c.pt"), "stability_initial_checkpoint_sha256": _sha256(output / "initial_from_c.pt"), "references": {"experiment_c": identity_c, "experiment_d_step0": identity_d}, "parameters_match": _sha256(c_checkpoint) == _sha256(d_output / "initial_from_c.pt") == _sha256(output / "initial_from_c.pt")}
    write_metrics(output / "step0_identity.json", identity)
    if identity["status"] != "PASS" or not identity["parameters_match"]:
        raise RuntimeError(f"D-Stability step0 identity failed: {identity}")
    preflight = _preflight_accounting(model, specs, device)
    write_metrics(output / "transaction_cost_preflight.json", preflight)
    (output / "transaction_cost_preflight.md").write_text("# Transaction Cost Preflight\n\nStatus: **%s**\n\nToy: `%s`\n\nReal: `%s`\n\nBatch boundary: `%s`\n\nGradient: `%s`\n" % (preflight["status"], preflight["toy_test"], preflight["real_train_slice"], preflight["batch_boundary_test"], preflight["gradient_test"]), encoding="utf-8")
    if preflight["status"] != "PASS" or preflight["batch_boundary_test"]["status"] != "PASS":
        raise RuntimeError("D-Stability transaction-cost preflight failed")
    decomposition = _step0_gradient_decomposition(model, specs, device)
    write_metrics(output / "step0_gradient_decomposition.json", decomposition)
    resolved = yaml.safe_load((d_output / "resolved_config.yaml").read_text(encoding="utf-8"))
    resolved["global_optimizer"]["lr"] = STABILITY_LR
    resolved["logging"]["output_dir"] = str(output)
    resolved["experiment_d_stability"] = {"initialization": "Experiment C best; identical to Experiment D step0", "source_experiment_d": str(d_output), "learning_rate": STABILITY_LR, "reference_d_learning_rate": REFERENCE_D_LR, "weight_decay": config.global_optimizer.weight_decay, "lambda_score": 1.0, "lambda_portfolio": LAMBDA_PORTFOLIO, "temperature": 1.0, "transaction_cost_bps": COST_BPS, "extra_turnover_penalty": 0.0, "validation_target": "validation_daily_sharpe_7_5bps", "validation_frequency_steps": 10, "early_stopping_patience_evaluations": config.training.early_stopping_patience, "no_2022": True, "output_dir": str(output)}
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")

    optimizer = torch.optim.AdamW(_mlp_parameters(model), lr=STABILITY_LR, weight_decay=config.global_optimizer.weight_decay)
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_metric = float(step0["metrics"]["validation_daily_sharpe_cost_7p5bps"])
    best_step = 0; step = 0; no_improve = 0; stopped_early = False
    history: list[dict[str, Any]] = [{"optimizer_step": 0, "epoch_fraction": 0.0, "learning_rate": STABILITY_LR, "train_score_loss": None, "train_net_loss": None, "train_total_loss": None, "train_rankic": None, "validation_rankic": step0["metrics"]["validation_execution_rankic"], "validation_annual_return_0bps": step0["metrics"]["validation_daily_return_cost_0p0bps"], "validation_sharpe_0bps": step0["metrics"]["validation_daily_sharpe_cost_0p0bps"], "validation_annual_return_7_5bps": step0["metrics"]["validation_daily_return_cost_7p5bps"], "validation_sharpe_7_5bps": step0["metrics"]["validation_daily_sharpe_cost_7p5bps"], "validation_mdd_7_5bps": step0["metrics"]["validation_daily_mdd_cost_7p5bps"], "validation_turnover": step0["metrics"]["validation_daily_turnover"], "validation_effective_n": step0["metrics"]["validation_effective_n"], "validation_max_weight": step0["metrics"]["validation_max_weight_mean"], "validation_information_ratio_7_5bps": step0["metrics"]["validation_information_ratio_cost_7p5bps"], "parameter_l2_distance": 0.0, "relative_parameter_l2_distance": 0.0, "pre_clip_gradient_norm": None, "post_clip_gradient_norm": None, "parameter_update_l2_norm": None, "relative_update_norm": None, "gradient_norm_total": None, "improved": True}]
    checkpoints = output / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    update_rows: list[dict[str, Any]] = []
    for epoch in range(config.training.epochs):
        for sequence, loss_start, batch_index in specs:
            batch = _collate_rows(sequence, device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            components = _net_components(model, batch, loss_start)
            total_loss = components["score_loss"] + LAMBDA_PORTFOLIO * components["net_loss"]
            if not torch.isfinite(total_loss):
                raise FloatingPointError("non-finite D-Stability total loss")
            total_loss.backward()
            pre_clip = _gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(_mlp_parameters(model), config.training.gradient_clip_norm, error_if_nonfinite=True)
            post_clip = _gradient_norm(model)
            before = _state_vector(model)
            optimizer.step(); step += 1
            update = _param_update_record(model, initial, step, step / len(specs), STABILITY_LR, pre_clip, post_clip, before)
            update_rows.append(update)
            if step % 10 != 0:
                continue
            model.eval()
            evaluation = _evaluate(model, loaders.validation, source, device, config)
            metric = float(evaluation["metrics"]["validation_daily_sharpe_cost_7p5bps"])
            improved = metric > best_metric + 1e-12
            no_improve = 0 if improved else no_improve + 1
            if improved:
                best_metric = metric; best_step = step; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            loss_scores = components["scores"][loss_start:].detach().cpu().numpy(); loss_returns = batch["execution_1d_return"][loss_start:].detach().cpu().numpy(); loss_masks = (batch["asset_mask"][loss_start:] & batch["execution_return_mask"][loss_start:]).detach().cpu().numpy(); rankics = [_rank_corr(loss_scores[i][loss_masks[i]], loss_returns[i][loss_masks[i]]) for i in range(len(loss_scores)) if int(loss_masks[i].sum()) >= 3]
            row = {"optimizer_step": step, "epoch_fraction": step / len(specs), "learning_rate": STABILITY_LR, "train_score_loss": float(components["score_loss"].detach().cpu()), "train_net_loss": float(components["net_loss"].detach().cpu()), "train_total_loss": float(total_loss.detach().cpu()), "train_rankic": float(np.mean(rankics)) if rankics else 0.0, "validation_rankic": evaluation["metrics"]["validation_execution_rankic"], "validation_annual_return_0bps": evaluation["metrics"]["validation_daily_return_cost_0p0bps"], "validation_sharpe_0bps": evaluation["metrics"]["validation_daily_sharpe_cost_0p0bps"], "validation_annual_return_7_5bps": evaluation["metrics"]["validation_daily_return_cost_7p5bps"], "validation_sharpe_7_5bps": evaluation["metrics"]["validation_daily_sharpe_cost_7p5bps"], "validation_mdd_7_5bps": evaluation["metrics"]["validation_daily_mdd_cost_7p5bps"], "validation_turnover": evaluation["metrics"]["validation_daily_turnover"], "validation_effective_n": evaluation["metrics"]["validation_effective_n"], "validation_max_weight": evaluation["metrics"]["validation_max_weight_mean"], "validation_information_ratio_7_5bps": evaluation["metrics"]["validation_information_ratio_cost_7p5bps"], "parameter_l2_distance": update["parameter_l2_distance"], "relative_parameter_l2_distance": update["relative_parameter_l2_distance"], "pre_clip_gradient_norm": pre_clip, "post_clip_gradient_norm": post_clip, "parameter_update_l2_norm": update["parameter_update_l2_norm"], "relative_update_norm": update["relative_update_norm"], "gradient_norm_total": pre_clip, "improved": improved}
            history.append(row)
            print(f"[D-Stability] step={step} epoch={row['epoch_fraction']:.3f} lr={STABILITY_LR:.1e} val_sharpe_7.5bp={metric:.6f} turnover={row['validation_turnover']:.5f} effN={row['validation_effective_n']:.2f} param_rel={row['relative_parameter_l2_distance']:.6f} update={row['parameter_update_l2_norm']:.6g} improved={improved} no_improve={no_improve}", flush=True)
            if improved:
                _save_checkpoint(checkpoints / "best.pt", model, optimizer, step, best_step, best_metric)
            if step in {10, 20, 50, 100}:
                _save_checkpoint(checkpoints / f"step{step}.pt", model, optimizer, step, best_step, best_metric)
            if no_improve >= config.training.early_stopping_patience:
                stopped_early = True; break
        if stopped_early:
            break
    _save_checkpoint(checkpoints / "last.pt", model, optimizer, step, best_step, best_metric)
    if not (checkpoints / "best.pt").is_file():
        model.load_state_dict(best_state); _save_checkpoint(checkpoints / "best.pt", model, optimizer, best_step, best_step, best_metric)
    model.load_state_dict(best_state)
    final = _evaluate(model, loaders.validation, source, device, config)
    final_metrics = dict(final["metrics"])
    final_metrics.update({"experiment": "experiment_d_stability", "training_performed": True, "learning_rate": STABILITY_LR, "reference_d_learning_rate": REFERENCE_D_LR, "best_optimizer_step": best_step, "best_epoch_fraction": best_step / len(specs), "optimizer_steps_total": step, "early_stopped": stopped_early, "early_stop_reason": "validation_7p5bp_sharpe_patience" if stopped_early else "max_epochs_or_steps", "lambda_portfolio": LAMBDA_PORTFOLIO, "lambda_score": 1.0, "temperature": 1.0, "transaction_cost_bps": COST_BPS, "no_extra_turnover_penalty": True, "no_topk_or_cap": True, "no_2022_data": True, "c_best_checkpoint_sha256": _sha256(c_checkpoint), "d_initial_checkpoint_sha256": _sha256(d_output / "initial_from_c.pt"), "stability_initial_checkpoint_sha256": _sha256(output / "initial_from_c.pt"), "best_checkpoint_sha256": _sha256(checkpoints / "best.pt"), "last_checkpoint_sha256": _sha256(checkpoints / "last.pt"), "validation_date_range": final["dates"], "validation_date_count": len(final["dates"]), "step0_identity_status": identity["status"], "preflight_status": preflight["status"]})
    final["score_frame"].to_parquet(output / "validation_scores.parquet", index=False)
    final["weight_frame"].to_parquet(output / "validation_weights.parquet", index=False)
    final["diagnostics"].to_csv(output / "validation_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    daily = _daily_with_costs(final["daily_returns"])
    stability, stability_stats = _stability_diagnostics(final["weight_frame"], final["dates"])
    final["diagnostics"].merge(stability, on="date", how="left").to_csv(output / "turnover_diagnostics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output / "validation_daily_returns.csv", index=False, encoding="utf-8-sig")
    final_metrics["weight_stability"] = stability_stats
    best_history = [row for row in history if int(row["optimizer_step"]) == best_step]
    final_metrics["train_rankic_at_best_training_batch"] = best_history[0].get("train_rankic") if best_history else None
    pd.DataFrame(history).to_csv(output / "training_steps.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(update_rows).to_csv(output / "parameter_distance.csv", index=False, encoding="utf-8-sig")
    write_metrics(output / "validation_metrics.json", final_metrics)

    # Full-period audit uses one identical evaluator for all six checkpoints.
    model = E2EAIModel(config.model, config.ablation).to(device)
    audit_rows: list[dict[str, Any]] = []
    for experiment, base, initial_path, best_path, last_path in [
        ("experiment_d", d_output, d_output / "initial_from_c.pt", d_output / "checkpoints" / "best.pt", d_output / "checkpoints" / "last.pt"),
        ("experiment_d_stability", output, output / "initial_from_c.pt", output / "checkpoints" / "best.pt", output / "checkpoints" / "last.pt"),
    ]:
        initial_vec = initial
        for label, path in [("initial_from_c.pt", initial_path), ("best.pt", best_path), ("last.pt", last_path)]:
            row = _audit_checkpoint(model, path, loaders.train, loaders.validation, device, initial_vec)
            row["experiment"] = experiment; row["checkpoint"] = label
            audit_rows.append(row)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output / "full_period_metric_audit.csv", index=False, encoding="utf-8-sig")
    reported_d_train = _load_metrics(d_output / "validation_metrics.json").get("train_rankic_at_best")
    audit_table = audit.to_csv(index=False)
    audit_note = ["# Full-Period RankIC Audit", "", "All checkpoints use the same full-period Spearman RankIC evaluator over complete train and validation decision dates.", "", "```csv", audit_table.rstrip(), "```", "", f"Original Experiment D reported training-log RankIC: {reported_d_train}", "The reported value is a best-step mini-batch/rolling training-log metric, not the full-period evaluator metric."]
    (output / "full_period_metric_audit.md").write_text("\n".join(audit_note) + "\n", encoding="utf-8")

    # Align the original D learning curve with the stability curve.
    d_history = pd.read_csv(d_output / "training_steps.csv")
    stability_history = pd.DataFrame(history)
    d_learning = d_history[["optimizer_step", "epoch_fraction", "validation_sharpe_7_5bps", "validation_annual_return_7_5bps", "validation_rankic", "validation_turnover", "validation_effective_n", "validation_information_ratio_7_5bps"]].copy()
    d_learning = d_learning.rename(columns={column: f"D_{column}" for column in d_learning.columns if column != "optimizer_step"})
    s_learning = stability_history[["optimizer_step", "epoch_fraction", "learning_rate", "validation_sharpe_7_5bps", "validation_annual_return_7_5bps", "validation_rankic", "validation_turnover", "validation_effective_n", "validation_information_ratio_7_5bps", "relative_parameter_l2_distance", "parameter_update_l2_norm"]].copy()
    s_learning = s_learning.rename(columns={column: f"D_stability_{column}" for column in s_learning.columns if column != "optimizer_step"})
    curve = d_learning.merge(s_learning, on="optimizer_step", how="outer").sort_values("optimizer_step")
    d_audit = audit[audit["experiment"] == "experiment_d"].set_index("checkpoint")
    known_d_distances = {0: float(d_audit.loc["initial_from_c.pt", "relative_parameter_l2_distance"]), int(_load_metrics(d_output / "validation_metrics.json")["best_optimizer_step"]): float(d_audit.loc["best.pt", "relative_parameter_l2_distance"]), int(_load_metrics(d_output / "validation_metrics.json")["optimizer_steps_total"]): float(d_audit.loc["last.pt", "relative_parameter_l2_distance"])}
    curve["D_relative_parameter_l2_distance"] = curve["optimizer_step"].map(known_d_distances)
    curve.to_csv(output / "D_vs_D_stability_learning_curve.csv", index=False, encoding="utf-8-sig")

    comparison, summary = _build_comparison(output, d_output, final_metrics, audit, d_history, stability_history, initial, model, initial)
    summary.update({"preflight_status": preflight["status"], "step0_identity_status": identity["status"], "reported_d_train_rankic": reported_d_train, "full_period_audit": audit.to_dict("records"), "only_training_variable_learning_rate": True, "d_learning_rate": REFERENCE_D_LR, "d_stability_learning_rate": STABILITY_LR, "weight_decay_same": True, "lambda_same": True, "temperature_same": True, "seed_same": True, "batch_logic_same": True, "no_2022": True, "experiment_e_run": False})
    write_metrics(output / "stability_summary.json", summary)
    manifest = {"experiment": "experiment_d_stability", "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_experiment_c": str(c_output), "source_experiment_d": str(d_output), "output_dir": str(output), "c_best_checkpoint_sha256": _sha256(c_checkpoint), "d_initial_checkpoint_sha256": _sha256(d_output / "initial_from_c.pt"), "stability_initial_checkpoint_sha256": _sha256(output / "initial_from_c.pt"), "initial_hashes_match": _sha256(c_checkpoint) == _sha256(d_output / "initial_from_c.pt") == _sha256(output / "initial_from_c.pt"), "step0_identity_status": identity["status"], "preflight_status": preflight["status"], "learning_rate": STABILITY_LR, "reference_d_learning_rate": REFERENCE_D_LR, "weight_decay": config.global_optimizer.weight_decay, "lambda_portfolio": LAMBDA_PORTFOLIO, "temperature": 1.0, "transaction_cost_bps": COST_BPS, "extra_turnover_penalty": 0.0, "no_2022": True, "experiment_e_run": False, "optimizer_steps_total": step, "best_optimizer_step": best_step, "early_stopped": stopped_early, "files": sorted({path.name for path in output.iterdir()} | {"run_manifest.json"})}
    write_metrics(output / "run_manifest.json", manifest)
    print(f"[D-Stability] completed steps={step} best_step={best_step} best_sharpe_7.5bp={best_metric:.6f} output={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--c-output", type=Path, required=True)
    parser.add_argument("--d-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        print(_finalize_existing(args.config, c_output=args.c_output, d_output=args.d_output, output=args.output))
    else:
        print(run(args.config, c_output=args.c_output, d_output=args.d_output, output=args.output))


if __name__ == "__main__":
    main()
