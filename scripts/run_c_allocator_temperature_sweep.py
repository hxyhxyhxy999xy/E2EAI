"""Evaluate Experiment C's frozen-score allocator at four temperatures.

The C best checkpoint is loaded once in inference mode.  Raw scores and the
cross-sectional z-scores are then held fixed while only the stable masked
softmax temperature changes.  No optimizer, backward pass, checkpoint write,
or validation search is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.evaluation.continuous_portfolio import continuous_long_only_weights
from e2eai.models.e2eai import E2EAIModel
from e2eai.utils.reproducibility import resolve_device, seed_everything
from e2eai.workflows import load_configured_data, write_metrics
from scripts.run_daily_experiment_b import (
    _benchmark_for_dates,
    _canonical_dates,
    _performance,
    _sha256,
    _token,
)
from scripts.run_daily_experiment_c import _collect_scores, _pearson, _rank_corr


TEMPERATURES: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)
COSTS: tuple[float, ...] = (0.0, 5.0, 7.5, 10.0)


def _temperature_label(value: float) -> str:
    return f"T_{value:.1f}"


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame[columns].copy().sort_values(columns[:2], kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode("utf-8"))
    for row in ordered.itertuples(index=False, name=None):
        for value in row:
            if isinstance(value, float):
                digest.update(np.float64(value).tobytes())
            else:
                digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _canonical_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "asset_id" in result.columns and "asset" not in result.columns:
        result["asset"] = result["asset_id"].astype(str)
    result["date"] = _canonical_dates(result["date"])
    result["asset"] = result["asset"].astype(str)
    result["raw_score"] = pd.to_numeric(result["raw_score"], errors="coerce")
    return result.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)


def _score_identity(formal: pd.DataFrame, regenerated: pd.DataFrame) -> dict[str, Any]:
    left = _canonical_score_frame(formal)
    right = _canonical_score_frame(regenerated)
    same_shape = left.shape == right.shape
    same_axis = bool(
        same_shape
        and np.array_equal(left[["date", "asset"]].astype(str).to_numpy(), right[["date", "asset"]].astype(str).to_numpy())
    )
    left_values = left["raw_score"].to_numpy(dtype=np.float64)
    right_values = right["raw_score"].to_numpy(dtype=np.float64)
    finite_match = bool(same_shape and np.array_equal(np.isfinite(left_values), np.isfinite(right_values)))
    both = np.isfinite(left_values) & np.isfinite(right_values)
    max_diff = float(np.max(np.abs(left_values[both] - right_values[both]))) if bool(both.any()) else 0.0
    return {
        "status": "PASS" if same_shape and same_axis and finite_match and max_diff <= 1e-6 else "FAIL",
        "shape_formal": list(left.shape),
        "shape_regenerated": list(right.shape),
        "same_date_asset_axis": same_axis,
        "finite_mask_match": finite_match,
        "raw_score_max_abs_diff": max_diff,
        "tolerance": 1e-6,
        "formal_raw_score_sha256": _frame_hash(left, ["date", "asset", "raw_score"]),
        "regenerated_raw_score_sha256": _frame_hash(right, ["date", "asset", "raw_score"]),
    }


def _build_fixed_score_inputs(score_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    frame = _canonical_score_frame(score_frame)
    records: list[dict[str, Any]] = []
    by_date: dict[str, dict[str, Any]] = {}
    dates: list[str] = []
    for date, group in frame.groupby("date", sort=True):
        raw = group["raw_score"].to_numpy(dtype=np.float64)
        valid = np.isfinite(raw)
        weights_1, z, mean, std = continuous_long_only_weights(raw, valid, temperature=1.0, eps=1e-6)
        assets = group["asset"].astype(str).tolist()
        execution = pd.to_numeric(group["execution_1d_return"], errors="coerce").to_numpy(dtype=np.float64)
        by_date[str(date)] = {
            "assets": assets,
            "raw": raw,
            "valid": valid,
            "z": z,
            "mean": mean,
            "std": std,
            "execution": execution,
            "weights_t1": weights_1,
        }
        dates.append(str(date))
        for asset, raw_value, z_value, ok, ret in zip(assets, raw, z, valid, execution):
            records.append(
                {
                    "date": str(date),
                    "asset": asset,
                    "raw_score": float(raw_value) if np.isfinite(raw_value) else np.nan,
                    "normalized_score": float(z_value) if ok else np.nan,
                    "valid_asset": bool(ok),
                    "execution_1d_return": float(ret) if np.isfinite(ret) else np.nan,
                }
            )
    return pd.DataFrame(records), by_date, dates


def _stable_temperature_weights(fixed: dict[str, Any], temperature: float) -> np.ndarray:
    if temperature not in TEMPERATURES:
        raise ValueError(f"temperature must be one of {TEMPERATURES}")
    z = np.asarray(fixed["z"], dtype=np.float64)
    valid = np.asarray(fixed["valid"], dtype=bool)
    logits = z[valid] / float(temperature)
    stable = logits - float(np.max(logits))
    exponentials = np.exp(stable)
    weights_valid = exponentials / float(exponentials.sum())
    weights = np.zeros_like(z, dtype=np.float64)
    weights[valid] = weights_valid
    if not np.isfinite(weights).all():
        raise FloatingPointError(f"non-finite weights at temperature={temperature}")
    return weights


def _target_weights_and_frame(
    fixed_by_date: dict[str, dict[str, Any]], dates: list[str], temperature: float
) -> tuple[list[dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    targets: list[dict[str, float]] = []
    weight_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    deciles: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    corr_maps: list[dict[str, float]] = []
    adjacent_pearson: list[float] = []
    adjacent_spearman: list[float] = []
    for date in dates:
        fixed = fixed_by_date[date]
        assets = fixed["assets"]
        raw = fixed["raw"]
        valid = fixed["valid"]
        z = fixed["z"]
        execution = fixed["execution"]
        weights = _stable_temperature_weights(fixed, temperature)
        target = {asset: float(weight) for asset, weight, ok in zip(assets, weights, valid) if ok}
        targets.append(target)
        positive = weights[valid]
        hhi = float(np.square(positive).sum())
        entropy = float(-np.sum(positive * np.log(np.maximum(positive, 1e-300))))
        normalized_entropy = float(entropy / np.log(len(positive))) if len(positive) > 1 else 0.0
        counts = {f"n_weight_gt_{threshold:g}pct": int((positive > threshold).sum()) for threshold in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)}
        diagnostics.append(
            {
                "date": date,
                "temperature": temperature,
                "valid_asset_count": int(valid.sum()),
                "score_mean": float(fixed["mean"]),
                "score_std": float(fixed["std"]),
                "effective_n": float(1.0 / max(hhi, 1e-12)),
                "hhi": hhi,
                "max_weight": float(positive.max()),
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                **counts,
            }
        )
        valid_indices = np.flatnonzero(valid)
        order = valid_indices[np.argsort(raw[valid_indices], kind="stable")]
        n = len(order)
        shares = {f"D{index}_weight_share": 0.0 for index in range(1, 11)}
        for decile in range(1, 11):
            selected = order[np.floor(np.arange(n) * 10 / n).astype(int) + 1 == decile] if n else np.asarray([], dtype=int)
            shares[f"D{decile}_weight_share"] = float(weights[selected].sum())
        deciles.append({"date": date, "temperature": temperature, **shares})
        top = {"date": date, "temperature": temperature}
        for count in (10, 25, 50, 100):
            top[f"top{count}_score_weight_share"] = float(weights[order[-min(count, n):]].sum()) if n else 0.0
        top_rows.append(top)
        current_map = {assets[index]: float(weights[index]) for index in valid_indices}
        if corr_maps:
            common = sorted(set(corr_maps[-1]) & set(current_map))
            if len(common) >= 2:
                previous = np.asarray([corr_maps[-1][asset] for asset in common], dtype=np.float64)
                current = np.asarray([current_map[asset] for asset in common], dtype=np.float64)
                adjacent_pearson.append(_pearson(previous, current))
                adjacent_spearman.append(_rank_corr(previous, current))
        corr_maps.append(current_map)
        for asset, raw_value, z_value, weight, ok, ret in zip(assets, raw, z, weights, valid, execution):
            weight_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "temperature": temperature,
                    "raw_score": float(raw_value) if np.isfinite(raw_value) else np.nan,
                    "normalized_score": float(z_value) if ok else np.nan,
                    "weight": float(weight),
                    "valid_asset": bool(ok),
                    "execution_1d_return": float(ret) if np.isfinite(ret) else np.nan,
                }
            )
    diagnostics_frame = pd.DataFrame(diagnostics)
    decile_frame = pd.DataFrame(deciles)
    top_frame = pd.DataFrame(top_rows)
    structure = {
        "weight_corr_pearson": float(np.mean(adjacent_pearson)) if adjacent_pearson else 0.0,
        "weight_corr_spearman": float(np.mean(adjacent_spearman)) if adjacent_spearman else 0.0,
    }
    return targets, pd.DataFrame(weight_rows), diagnostics_frame, decile_frame, top_frame, structure


def _execution_returns(score_frame: pd.DataFrame, dates: list[str]) -> list[dict[str, float]]:
    values: list[dict[str, float]] = []
    for date, group in score_frame.groupby("date", sort=True):
        if str(date) not in dates:
            continue
        values.append({str(asset): float(value) for asset, value in zip(group["asset"], pd.to_numeric(group["execution_1d_return"], errors="coerce")) if np.isfinite(value)})
    return values


def _rankic_and_pearson(score_frame: pd.DataFrame) -> tuple[float, float]:
    rankics: list[float] = []
    pears: list[float] = []
    for _, group in score_frame.groupby("date", sort=True):
        raw = group["raw_score"].to_numpy(dtype=np.float64)
        ret = pd.to_numeric(group["execution_1d_return"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(raw) & np.isfinite(ret)
        if valid.sum() >= 3:
            rankics.append(_rank_corr(raw[valid], ret[valid]))
            pears.append(_pearson(raw[valid], ret[valid]))
    return (float(np.mean(rankics)) if rankics else 0.0, float(np.mean(pears)) if pears else 0.0)


def _daily_metric_frame(
    dates: list[str], daily: pd.DataFrame, diagnostics: pd.DataFrame, deciles: pd.DataFrame, top: pd.DataFrame
) -> pd.DataFrame:
    result = daily.copy()
    result.insert(0, "date", dates)
    for extra in (diagnostics, deciles, top):
        result = result.merge(extra, on="date", how="left")
    return result


def _summary_row(
    temperature: float,
    metrics: dict[str, Any],
    daily: pd.DataFrame,
    diagnostics: pd.DataFrame,
    deciles: pd.DataFrame,
    top: pd.DataFrame,
    structure: dict[str, float],
    rankic: float,
    pearson: float,
) -> dict[str, Any]:
    def q(column: str, value: float) -> float:
        return float(diagnostics[column].quantile(value))

    row: dict[str, Any] = {
        "temperature": temperature,
        "validation_rankic": rankic,
        "validation_pearson_ic": pearson,
        "annual_return_0bps": metrics["validation_daily_return_cost_0p0bps"],
        "sharpe_0bps": metrics["validation_daily_sharpe_cost_0p0bps"],
        "mdd_0bps": metrics["validation_daily_mdd_cost_0p0bps"],
        "annual_return_5bps": metrics["validation_daily_return_cost_5p0bps"],
        "sharpe_5bps": metrics["validation_daily_sharpe_cost_5p0bps"],
        "annual_return_7_5bps": metrics["validation_daily_return_cost_7p5bps"],
        "sharpe_7_5bps": metrics["validation_daily_sharpe_cost_7p5bps"],
        "mdd_7_5bps": metrics["validation_daily_mdd_cost_7p5bps"],
        "annual_return_10bps": metrics["validation_daily_return_cost_10p0bps"],
        "sharpe_10bps": metrics["validation_daily_sharpe_cost_10p0bps"],
        "turnover_mean": float(daily["one_way_turnover"].mean()),
        "turnover_median": float(daily["one_way_turnover"].median()),
        "turnover_p90": float(daily["one_way_turnover"].quantile(0.90)),
        "turnover_p95": float(daily["one_way_turnover"].quantile(0.95)),
        "turnover_p99": float(daily["one_way_turnover"].quantile(0.99)),
        "turnover_max": float(daily["one_way_turnover"].max()),
        "effective_n_mean": float(diagnostics["effective_n"].mean()),
        "effective_n_median": float(diagnostics["effective_n"].median()),
        "effective_n_p10": q("effective_n", 0.10),
        "effective_n_min": float(diagnostics["effective_n"].min()),
        "max_weight_mean": float(diagnostics["max_weight"].mean()),
        "max_weight_median": float(diagnostics["max_weight"].median()),
        "max_weight_p95": q("max_weight", 0.95),
        "max_weight_absolute": float(diagnostics["max_weight"].max()),
        "hhi_mean": float(diagnostics["hhi"].mean()),
        "normalized_entropy_mean": float(diagnostics["normalized_entropy"].mean()),
        "normalized_entropy_median": float(diagnostics["normalized_entropy"].median()),
        "D10_weight_share": float(deciles["D10_weight_share"].mean()),
        "D9_D10_weight_share": float((deciles["D9_weight_share"] + deciles["D10_weight_share"]).mean()),
        "D8_D10_weight_share": float((deciles["D8_weight_share"] + deciles["D9_weight_share"] + deciles["D10_weight_share"]).mean()),
        "D1_weight_share": float(deciles["D1_weight_share"].mean()),
        "top10_score_weight_share": float(top["top10_score_weight_share"].mean()),
        "top25_score_weight_share": float(top["top25_score_weight_share"].mean()),
        "top50_score_weight_share": float(top["top50_score_weight_share"].mean()),
        "top100_score_weight_share": float(top["top100_score_weight_share"].mean()),
        "weight_corr_pearson": structure["weight_corr_pearson"],
        "weight_corr_spearman": structure["weight_corr_spearman"],
        "annual_excess_return_7_5bps": metrics["validation_active_annual_excess_return_cost_7p5bps"],
        "tracking_error_7_5bps": metrics["validation_tracking_error_cost_7p5bps"],
        "information_ratio_7_5bps": metrics["validation_information_ratio_cost_7p5bps"],
        "active_mdd_7_5bps": metrics["validation_active_max_drawdown_cost_7p5bps"],
        "active_win_rate_7_5bps": metrics["validation_active_win_rate_cost_7p5bps"],
        "active_cumulative_return_7_5bps": metrics["validation_active_cumulative_return_cost_7p5bps"],
        "mean_n_weight_gt_0_1pct": float(diagnostics["n_weight_gt_0.001pct"].mean()),
        "mean_n_weight_gt_0_25pct": float(diagnostics["n_weight_gt_0.0025pct"].mean()),
        "mean_n_weight_gt_0_5pct": float(diagnostics["n_weight_gt_0.005pct"].mean()),
        "mean_n_weight_gt_1pct": float(diagnostics["n_weight_gt_0.01pct"].mean()),
        "mean_n_weight_gt_2pct": float(diagnostics["n_weight_gt_0.02pct"].mean()),
        "mean_n_weight_gt_5pct": float(diagnostics["n_weight_gt_0.05pct"].mean()),
    }
    row["concentration_warning"] = "HIGH_CONCENTRATION_WARNING" if row["max_weight_absolute"] > 0.10 else None
    if row["effective_n_mean"] < 30:
        row["concentration_warning"] = "LOW_EFFECTIVE_N_WARNING"
    if row["max_weight_absolute"] > 0.20:
        row["concentration_warning"] = "SEVERE_CONCENTRATION_WARNING"
    return row


def _write_summary_md(path: Path, summary: pd.DataFrame, incremental: pd.DataFrame, identity: dict[str, Any], warnings: list[str]) -> None:
    columns = ["temperature", "annual_return_0bps", "sharpe_0bps", "annual_return_7_5bps", "sharpe_7_5bps", "turnover_mean", "effective_n_mean", "max_weight_mean", "D10_weight_share", "information_ratio_7_5bps"]
    lines = [
        "# Experiment C Allocator Frozen-Score Temperature Sensitivity",
        "",
        "This is a four-point 2021 validation sensitivity evaluation with Experiment C best raw scores frozen. It is not a final OOS selection and does not run Experiment D.",
        "",
        f"- T=1 identity status: **{identity['status']}**; weight max diff={identity['weight_max_abs_diff']:.3g}; 7.5bp net-return max diff={identity['net_return_7p5bps_max_abs_diff']:.3g}.",
        f"- Shared raw-score hash: `{identity['raw_score_sha256']}`; shared z-score hash: `{identity['zscore_sha256']}`.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in summary[columns].iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    lines.extend(["", "## Incremental economics vs T=1", "", incremental.to_csv(index=False), ""])
    if warnings:
        lines.extend(["## Warnings", "", *[f"- {warning}" for warning in warnings], ""])
    else:
        lines.extend(["## Warnings", "", "- None of the predefined concentration warnings triggered.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _identity_against_c(
    t1_weights: pd.DataFrame,
    t1_daily: pd.DataFrame,
    c_output: Path,
    generated_scores: pd.DataFrame,
) -> dict[str, Any]:
    c_weights = pd.read_parquet(c_output / "validation_weights.parquet")
    c_weights = c_weights.rename(columns={"asset_id": "asset"}) if "asset_id" in c_weights.columns else c_weights
    left = c_weights[["date", "asset", "raw_score", "normalized_score", "weight", "valid_asset"]].copy()
    right = t1_weights[["date", "asset", "raw_score", "normalized_score", "weight", "valid_asset"]].copy()
    for frame in (left, right):
        frame["date"] = _canonical_dates(frame["date"])
        frame["asset"] = frame["asset"].astype(str)
    left = left.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    right = right.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    diffs = {}
    for column in ["raw_score", "normalized_score", "weight"]:
        diffs[f"{column}_max_abs_diff"] = float(np.max(np.abs(left[column].to_numpy(dtype=float) - right[column].to_numpy(dtype=float))))
    c_daily = pd.read_csv(c_output / "validation_daily_returns.csv")
    c_daily["date"] = _canonical_dates(c_daily["date"])
    ours = t1_daily.copy()
    ours["date"] = _canonical_dates(ours["date"])
    merged = c_daily.merge(ours, on="date", suffixes=("_c", "_sweep"), how="outer", validate="one_to_one")
    daily_columns = ["gross_return", "one_way_turnover", "net_return_0p0bps", "net_return_5p0bps", "net_return_7p5bps", "net_return_10p0bps"]
    daily_diffs = {f"{column}_max_abs_diff": float(np.max(np.abs(merged[f"{column}_c"].to_numpy(dtype=float) - merged[f"{column}_sweep"].to_numpy(dtype=float)))) for column in daily_columns}
    passed = bool(
        left.shape == right.shape
        and left[["date", "asset"]].astype(str).equals(right[["date", "asset"]].astype(str))
        and all(value <= 1e-6 for value in diffs.values())
        and all(value <= 1e-6 for value in daily_diffs.values())
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "c_best_checkpoint_sha256": _sha256(c_output / "checkpoints" / "best.pt"),
        "checkpoint_optimizer_step": 20,
        "weights_shape_c": list(left.shape),
        "weights_shape_temperature_1": list(right.shape),
        "same_date_asset_axis": bool(left[["date", "asset"]].astype(str).equals(right[["date", "asset"]].astype(str))),
        **diffs,
        **daily_diffs,
        "tolerance": 1e-6,
        "raw_score_sha256": _frame_hash(_canonical_score_frame(generated_scores), ["date", "asset", "raw_score"]),
    }
    return result


def run(config_path: Path, *, c_output: Path, output: Path | None = None) -> Path:
    c_output = c_output.resolve()
    output = (output or c_output.parent / "experiment_c_allocator").resolve()
    checkpoint = c_output / "checkpoints" / "best.pt"
    c_scores_path = c_output / "validation_scores.parquet"
    if not checkpoint.is_file() or not c_scores_path.is_file():
        raise FileNotFoundError("Experiment C best checkpoint and validation_scores.parquet are required")
    config = load_config(config_path)
    if config.model.horizons != [2] or config.model.execution_horizon != 2:
        raise ValueError("Temperature sweep requires h=2")
    if config.data.test_start is not None or config.data.test_end is not None:
        raise ValueError("Temperature sweep must not configure a test range")
    if tuple(TEMPERATURES) != (1.0, 0.8, 0.6, 0.4):
        raise RuntimeError("Temperature registration changed")
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    checkpoint_sha_before = _sha256(checkpoint)
    seed_everything(config.seed)
    source, synthetic = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    device = resolve_device(config.device)
    model = E2EAIModel(config.model, config.ablation).to(device)
    payload = __import__("torch").load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with __import__("torch").no_grad():
        regenerated_scores = _collect_scores(model, loaders.validation, device)
    regenerated_scores["date"] = _canonical_dates(regenerated_scores["date"])
    if regenerated_scores["date"].min() < "2021-01-01" or regenerated_scores["date"].max() > "2021-12-29":
        raise RuntimeError("Temperature sweep validation dates are outside 2021")
    formal_scores = pd.read_parquet(c_scores_path)
    raw_identity = _score_identity(formal_scores, regenerated_scores)
    if raw_identity["status"] != "PASS":
        raise RuntimeError(f"C best raw-score identity failed: {raw_identity}")
    fixed_frame, fixed_by_date, dates = _build_fixed_score_inputs(regenerated_scores)
    raw_hash = _frame_hash(fixed_frame, ["date", "asset", "raw_score"])
    z_hash = _frame_hash(fixed_frame, ["date", "asset", "normalized_score"])
    raw_score_identity = {
        "status": "PASS",
        "raw_score_sha256": raw_hash,
        "formal_c_raw_score_sha256": raw_identity["formal_raw_score_sha256"],
        "raw_score_max_abs_diff_vs_c_file": raw_identity["raw_score_max_abs_diff"],
        "zscore_sha256": z_hash,
        "temperatures": list(TEMPERATURES),
        "shared_raw_score_for_all_temperatures": True,
        "shared_zscore_for_all_temperatures": True,
        "validation_date_range": [dates[0], dates[-1]],
        "validation_date_count": len(dates),
        "used_2022": False,
    }
    write_metrics(output / "raw_score_identity.json", raw_score_identity)
    execution_returns = _execution_returns(regenerated_scores.assign(asset=regenerated_scores["asset_id"].astype(str)), dates)
    benchmark = _benchmark_for_dates(source, dates)
    rankic, pearson = _rankic_and_pearson(regenerated_scores)
    summary_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[pd.DataFrame] = []
    decile_rows: list[pd.DataFrame] = []
    top_rows: list[pd.DataFrame] = []
    per_temperature_metrics: dict[float, dict[str, Any]] = {}
    t1_identity: dict[str, Any] | None = None
    for temperature in TEMPERATURES:
        label = _temperature_label(temperature)
        destination = output / label
        destination.mkdir(parents=True, exist_ok=True)
        targets, weight_frame, diagnostics, deciles, top, structure = _target_weights_and_frame(fixed_by_date, dates, temperature)
        performance, daily = _performance(targets, execution_returns, benchmark, list(COSTS), annual_trading_days=config.evaluation.annual_trading_days, holding_threshold=config.evaluation.holding_threshold)
        daily_frame = _daily_metric_frame(dates, daily, diagnostics, deciles, top)
        metrics = _summary_row(temperature, performance, daily, diagnostics, deciles, top, structure, rankic, pearson)
        metrics.update(
            {
                "temperature": temperature,
                "experiment": "experiment_c_allocator",
                "mapping": "frozen_c_score_zscore_masked_softmax",
                "training_performed": False,
                "optimizer_steps": 0,
                "checkpoint_sha256": checkpoint_sha_before,
                "raw_score_sha256": raw_hash,
                "zscore_sha256": z_hash,
                "no_2022_data": True,
                "cost_scenarios_bps": list(COSTS),
                "daily_path_metrics": performance,
            }
        )
        per_temperature_metrics[temperature] = metrics
        summary_rows.append(metrics)
        daily_rows.append(daily_frame.assign(temperature=temperature))
        diagnostics_rows.append(diagnostics)
        decile_rows.append(deciles)
        top_rows.append(top)
        weight_frame.to_parquet(destination / "validation_weights.parquet", index=False)
        daily_frame.to_csv(destination / "validation_daily_returns.csv", index=False, encoding="utf-8-sig")
        diagnostics.to_csv(destination / "validation_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
        write_metrics(destination / "validation_metrics.json", metrics)
        if temperature == 1.0:
            t1_identity = _identity_against_c(weight_frame, daily_frame, c_output, regenerated_scores)
            write_metrics(output / "t1_vs_c_identity.json", t1_identity)
            if t1_identity["status"] != "PASS":
                raise RuntimeError(f"T=1 failed Experiment C identity: {t1_identity}")
    summary = pd.DataFrame(summary_rows).sort_values("temperature", ascending=False).reset_index(drop=True)
    summary.drop(columns=["daily_path_metrics"], errors="ignore").to_csv(output / "temperature_summary.csv", index=False, encoding="utf-8-sig")
    all_daily = pd.concat(daily_rows, ignore_index=True)
    all_diag = pd.concat(diagnostics_rows, ignore_index=True)
    all_deciles = pd.concat(decile_rows, ignore_index=True)
    all_top = pd.concat(top_rows, ignore_index=True)
    all_daily.to_csv(output / "temperature_daily_metrics.csv", index=False, encoding="utf-8-sig")
    all_diag.to_csv(output / "temperature_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    all_deciles.to_csv(output / "temperature_score_decile_weights.csv", index=False, encoding="utf-8-sig")
    all_top.to_csv(output / "temperature_top_score_weight_share.csv", index=False, encoding="utf-8-sig")
    benchmark_rows = []
    for row in summary_rows:
        benchmark_rows.append({"temperature": row["temperature"], "benchmark_annual_return_7_5bps": row["daily_path_metrics"]["validation_benchmark_annual_return_cost_7p5bps"], "strategy_annual_return_7_5bps": row["annual_return_7_5bps"], "annual_excess_return_7_5bps": row["annual_excess_return_7_5bps"], "tracking_error_7_5bps": row["tracking_error_7_5bps"], "information_ratio_7_5bps": row["information_ratio_7_5bps"], "active_max_drawdown_7_5bps": row["active_mdd_7_5bps"], "active_win_rate_7_5bps": row["active_win_rate_7_5bps"], "active_cumulative_return_7_5bps": row["active_cumulative_return_7_5bps"]})
    benchmark_frame = pd.DataFrame(benchmark_rows)
    benchmark_frame.to_csv(output / "temperature_benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    baseline = summary.loc[summary["temperature"] == 1.0].iloc[0]
    incremental_rows = []
    for _, row in summary[summary["temperature"] != 1.0].iterrows():
        gross_drag_t = float(row["annual_return_0bps"] - row["annual_return_7_5bps"])
        gross_drag_base = float(baseline["annual_return_0bps"] - baseline["annual_return_7_5bps"])
        delta_gross = float(row["annual_return_0bps"] - baseline["annual_return_0bps"])
        delta_net = float(row["annual_return_7_5bps"] - baseline["annual_return_7_5bps"])
        delta_drag = gross_drag_t - gross_drag_base
        incremental_rows.append({"temperature": row["temperature"], "delta_gross_annual_return": delta_gross, "delta_7_5bp_annual_return": delta_net, "delta_gross_sharpe": float(row["sharpe_0bps"] - baseline["sharpe_0bps"]), "delta_7_5bp_sharpe": float(row["sharpe_7_5bps"] - baseline["sharpe_7_5bps"]), "delta_turnover": float(row["turnover_mean"] - baseline["turnover_mean"]), "delta_effective_n": float(row["effective_n_mean"] - baseline["effective_n_mean"]), "delta_max_weight": float(row["max_weight_mean"] - baseline["max_weight_mean"]), "delta_information_ratio": float(row["information_ratio_7_5bps"] - baseline["information_ratio_7_5bps"]), "incremental_transaction_cost_drag": delta_drag, "incremental_gross_minus_cost_drag": delta_gross - delta_drag})
    incremental = pd.DataFrame(incremental_rows)
    incremental.to_csv(output / "temperature_incremental_analysis.csv", index=False, encoding="utf-8-sig")
    warnings = [f"{row['temperature']}: {row['concentration_warning']}" for row in summary_rows if row.get("concentration_warning")]
    _write_summary_md(output / "temperature_summary.md", summary, incremental, {**t1_identity, "raw_score_sha256": raw_hash, "zscore_sha256": z_hash}, warnings)
    resolved = yaml.safe_load((c_output / "resolved_config.yaml").read_text(encoding="utf-8"))
    resolved["allocator_temperature_sensitivity"] = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha_before, "optimizer_step": 20, "temperatures": list(TEMPERATURES), "only_allocator_variable": "temperature", "training": False, "backward": False, "optimizer_step_called": False, "no_2022": True}
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    checkpoint_sha_after = _sha256(checkpoint)
    manifest = {"experiment": "experiment_c_allocator_temperature_sweep", "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_experiment_c": str(c_output), "output_dir": str(output), "temperatures": list(TEMPERATURES), "checkpoint": str(checkpoint), "checkpoint_optimizer_step": 20, "checkpoint_sha256_before": checkpoint_sha_before, "checkpoint_sha256_after": checkpoint_sha_after, "checkpoint_unchanged": checkpoint_sha_before == checkpoint_sha_after, "training_performed": False, "backward_called": False, "optimizer_step_called": False, "lambda_recomputed": False, "allocator_changed": False, "zscore_changed": False, "other_temperatures_tested": False, "experiment_d_run": False, "used_2022": False, "validation_date_range": [dates[0], dates[-1]], "files": sorted(path.name for path in output.iterdir()) + ["run_manifest.json"]}
    write_metrics(output / "run_manifest.json", manifest)
    print(f"[allocator-sweep] T=1 identity PASS; temperatures={list(TEMPERATURES)} output={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--c-output", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(run(args.config, c_output=args.c_output, output=args.output))


if __name__ == "__main__":
    main()
