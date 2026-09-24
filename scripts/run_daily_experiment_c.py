"""Run Experiment C: gross-return end-to-end training through B's allocator.

The model starts from the frozen Experiment A checkpoint.  The only trainable
path is the SimpleMLP scorer -> differentiable cross-sectional z-score ->
temperature-1 masked softmax -> one-day gross execution return.  Transaction
costs and turnover are evaluation-only diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
from e2eai.evaluation.continuous_portfolio import (
    continuous_long_only_weights_torch_batch,
)
from e2eai.evaluation.daily_validation import _simulate_daily_weight_path
from e2eai.evaluation.metrics import maximum_drawdown
from e2eai.models.e2eai import E2EAIModel
from e2eai.training.losses import masked_cross_sectional_ic
from e2eai.utils.reproducibility import seed_everything, resolve_device
from e2eai.workflows import load_configured_data, write_metrics

from scripts.run_daily_experiment_b import (
    _benchmark_for_dates,
    _build_weights,
    _canonical_dates,
    _identity_frame,
    _performance,
    _sha256,
    _token,
)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _raw_scores(model: E2EAIModel, batch: dict[str, Any]) -> torch.Tensor:
    if model.simple_mlp_scorer is None:
        raise RuntimeError("Experiment C requires the SimpleMLP scorer")
    factors = batch["raw_factors"]
    mask = batch["asset_mask"].bool()
    clean = torch.where(mask.unsqueeze(-1) & torch.isfinite(factors), factors, torch.zeros_like(factors))
    hidden = model.simple_mlp_scorer.encode(clean, mask)
    return model.simple_mlp_scorer.score(hidden, mask)


def _valid_execution(batch: dict[str, Any], scores: torch.Tensor) -> torch.Tensor:
    execution = batch.get("execution_1d_return")
    execution_mask = batch.get("execution_return_mask")
    if not isinstance(execution, torch.Tensor) or not isinstance(execution_mask, torch.Tensor):
        raise ValueError("Experiment C requires execution returns and masks")
    return batch["asset_mask"].bool() & execution_mask.bool() & torch.isfinite(scores) & torch.isfinite(execution)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    clean = torch.where(mask, value, torch.zeros_like(value))
    count = mask.sum()
    return torch.where(count > 0, clean.sum() / count.to(value.dtype), value.sum() * 0.0)


def _score_loss(scores: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    execution = batch["execution_1d_return"]
    valid = _valid_execution(batch, scores)
    ic, estimable = masked_cross_sectional_ic(
        scores.unsqueeze(1), execution.unsqueeze(1), valid.unsqueeze(1), min_observations=3
    )
    loss = -_masked_mean(ic.squeeze(1), estimable.squeeze(1))
    return loss, ic.squeeze(1)


def _gross_loss(
    scores: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    execution = batch["execution_1d_return"]
    asset_mask = batch["asset_mask"].bool()
    weights, normalized, means, stds = continuous_long_only_weights_torch_batch(
        scores, asset_mask & torch.isfinite(scores), temperature=1.0, eps=1e-6
    )
    valid = _valid_execution(batch, scores)
    observed_weight = torch.where(valid, weights, torch.zeros_like(weights)).sum(dim=-1)
    weighted = torch.where(valid, weights * execution, torch.zeros_like(execution)).sum(dim=-1)
    gross = torch.where(observed_weight > 1e-6, weighted / observed_weight.clamp_min(1e-6), torch.zeros_like(weighted))
    valid_rows = observed_weight > 1e-6
    return -_masked_mean(gross, valid_rows), gross, normalized, weights


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum().cpu())
    return float(math.sqrt(total))


def _mlp_parameters(model: E2EAIModel) -> list[torch.nn.Parameter]:
    if model.simple_mlp_scorer is None:
        raise RuntimeError("SimpleMLP scorer is missing")
    return [parameter for parameter in model.simple_mlp_scorer.parameters() if parameter.requires_grad]


def _component_gradients(model: E2EAIModel, batch: dict[str, Any]) -> tuple[float, float]:
    parameters = _mlp_parameters(model)
    model.zero_grad(set_to_none=True)
    scores = _raw_scores(model, batch)
    score_loss, _ = _score_loss(scores, batch)
    score_loss.backward()
    score_norm = _grad_norm(parameters)
    model.zero_grad(set_to_none=True)
    scores = _raw_scores(model, batch)
    gross_loss, _, _, _ = _gross_loss(scores, batch)
    gross_loss.backward()
    gross_norm = _grad_norm(parameters)
    model.zero_grad(set_to_none=True)
    return score_norm, gross_norm


def _calibrate(model: E2EAIModel, loader: Any, device: torch.device, max_batches: int = 16) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(itertools.islice(loader, max_batches)):
        batch = _move_batch(dict(raw_batch), device)
        score_norm, gross_norm = _component_gradients(model, batch)
        rows.append(
            {
                "batch_index": batch_index,
                "batch_dates": int(batch["raw_factors"].shape[0]),
                "g_score": score_norm,
                "g_gross": gross_norm,
            }
        )
    if not rows:
        raise RuntimeError("No train batches were available for gradient calibration")
    score_values = np.asarray([row["g_score"] for row in rows], dtype=np.float64)
    gross_values = np.asarray([row["g_gross"] for row in rows], dtype=np.float64)
    median_score = float(np.median(score_values))
    median_gross = float(np.median(gross_values))
    if not np.isfinite(median_score) or median_score <= 0 or not np.isfinite(median_gross) or median_gross <= 0:
        raise RuntimeError(f"gradient scales suspicious: G_score={median_score}, G_gross={median_gross}")
    lambda_portfolio = median_score / median_gross
    if not np.isfinite(lambda_portfolio) or lambda_portfolio <= 0 or lambda_portfolio > 1e6 or lambda_portfolio < 1e-6:
        raise RuntimeError(f"gradient scales suspicious: lambda_portfolio={lambda_portfolio}")
    return {
        "calibration_data": "train split only",
        "batches_used": len(rows),
        "median_g_score": median_score,
        "median_g_gross": median_gross,
        "lambda_score": 1.0,
        "lambda_portfolio": float(lambda_portfolio),
        "batch_gradients": rows,
        "validation_used": False,
        "used_2021": False,
        "used_2022": False,
    }


@torch.no_grad()
def _collect_scores(model: E2EAIModel, loader: Any, device: torch.device) -> pd.DataFrame:
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        for raw_batch in loader:
            batch = _move_batch(dict(raw_batch), device)
            scores = _raw_scores(model, batch).detach().cpu()
            execution = batch.get("execution_1d_return")
            execution_mask = batch.get("execution_return_mask")
            identifiers = batch["asset_ids"]
            dates = batch["dates"]
            for row_index, (date, assets) in enumerate(zip(dates, identifiers)):
                for position, asset in enumerate(assets):
                    if not bool(batch["asset_mask"][row_index, position]):
                        continue
                    valid_return = isinstance(execution, torch.Tensor) and isinstance(execution_mask, torch.Tensor) and bool(execution_mask[row_index, position])
                    rows.append(
                        {
                            "date": str(date),
                            "asset_id": str(asset),
                            "raw_score": float(scores[row_index, position]),
                            "execution_1d_return": float(execution[row_index, position].detach().cpu()) if valid_return else np.nan,
                        }
                    )
    finally:
        model.train(was_training)
    if not rows:
        raise RuntimeError("No validation scores were collected")
    return pd.DataFrame(rows)


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _decile_frame(weight_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for date, group in weight_frame.groupby("date", sort=True):
        valid = group[group["valid_asset"] & np.isfinite(group["raw_score"])].copy()
        valid = valid.sort_values(["raw_score", "asset"], kind="stable").reset_index(drop=True)
        n = len(valid)
        shares = {f"weight_share_decile_{index}": 0.0 for index in range(1, 11)}
        if n:
            deciles = np.minimum(10, np.floor(np.arange(n) * 10 / n).astype(int) + 1)
            for decile in range(1, 11):
                shares[f"weight_share_decile_{decile}"] = float(valid.loc[deciles == decile, "weight"].sum())
        rows.append({"date": date, **shares})
    frame = pd.DataFrame(rows)
    means = {column: float(frame[column].mean()) for column in frame.columns if column.startswith("weight_share_decile_")}
    means["D10_weight_share"] = means.get("weight_share_decile_10", 0.0)
    means["D9_D10_weight_share"] = means.get("weight_share_decile_9", 0.0) + means.get("weight_share_decile_10", 0.0)
    means["D1_weight_share"] = means.get("weight_share_decile_1", 0.0)
    return frame, means


def _structure_features(weight_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    deciles, decile_means = _decile_frame(weight_frame)
    by_date: list[dict[str, Any]] = []
    weight_maps: list[dict[str, float]] = []
    for date, group in weight_frame.groupby("date", sort=True):
        valid = group[group["valid_asset"]].copy()
        raw = valid["raw_score"].to_numpy(dtype=np.float64)
        weights = valid["weight"].to_numpy(dtype=np.float64)
        normalized = valid["normalized_score"].to_numpy(dtype=np.float64)
        spearman = _rank_corr(raw, weights)
        pearson = _pearson(normalized, weights)
        weight_maps.append(dict(zip(valid["asset"].astype(str), weights)))
        by_date.append({"date": date, "score_weight_spearman": spearman, "normalized_weight_pearson": pearson})
    corr_frame = pd.DataFrame(by_date)
    stability_rows: list[dict[str, Any]] = []
    for index in range(1, len(weight_maps)):
        common = sorted(set(weight_maps[index - 1]) & set(weight_maps[index]))
        if len(common) < 2:
            continue
        previous = np.asarray([weight_maps[index - 1][asset] for asset in common], dtype=np.float64)
        current = np.asarray([weight_maps[index][asset] for asset in common], dtype=np.float64)
        stability_rows.append(
            {
                "date": str(weight_frame["date"].drop_duplicates().sort_values().iloc[index]),
                "weight_stability_pearson": _pearson(previous, current),
                "weight_stability_spearman": _rank_corr(previous, current),
                "common_assets": len(common),
            }
        )
    stability = pd.DataFrame(stability_rows)
    corr_stats: dict[str, float] = {}
    for column in ["score_weight_spearman", "normalized_weight_pearson"]:
        series = corr_frame[column].to_numpy(dtype=np.float64)
        corr_stats[f"{column}_mean"] = float(np.mean(series))
        corr_stats[f"{column}_median"] = float(np.median(series))
        corr_stats[f"{column}_p05"] = float(np.quantile(series, 0.05))
        corr_stats[f"{column}_p95"] = float(np.quantile(series, 0.95))
    for column in ["weight_stability_pearson", "weight_stability_spearman"]:
        series = stability[column].to_numpy(dtype=np.float64) if not stability.empty else np.asarray([0.0])
        corr_stats[f"{column}_mean"] = float(np.mean(series))
        corr_stats[f"{column}_median"] = float(np.median(series))
        corr_stats[f"{column}_p05"] = float(np.quantile(series, 0.05))
        corr_stats[f"{column}_p95"] = float(np.quantile(series, 0.95))
    features = corr_frame.merge(stability, on="date", how="left") if not stability.empty else corr_frame
    return deciles, {**decile_means, **corr_stats}


def _membership_decomposition(
    dates: list[str],
    weight_frame: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sets = [set(group.loc[group["valid_asset"], "asset"].astype(str)) for _, group in weight_frame.groupby("date", sort=True)]
    rows: list[dict[str, Any]] = []
    turnover = daily_returns["one_way_turnover"].to_numpy(dtype=np.float64)
    for index, date in enumerate(dates):
        previous = sets[index - 1] if index else set()
        current = sets[index]
        added = len(current - previous) if index else 0
        removed = len(previous - current) if index else 0
        changed = bool(index and (added or removed))
        rows.append(
            {
                "date": date,
                "turnover": float(turnover[index]),
                "membership_change_flag": changed,
                "number_added": added,
                "number_removed": removed,
            }
        )
    frame = pd.DataFrame(rows)
    normal = frame.loc[~frame["membership_change_flag"], "turnover"]
    changed = frame.loc[frame["membership_change_flag"], "turnover"]
    summary = {
        "mean_turnover_all_days": float(frame["turnover"].mean()),
        "mean_turnover_normal_days": float(normal.mean()) if len(normal) else 0.0,
        "mean_turnover_membership_change_days": float(changed.mean()) if len(changed) else 0.0,
        "membership_change_day_count": int(len(changed)),
        "top10_turnover_dates": frame.nlargest(10, "turnover").to_dict("records"),
    }
    return frame, summary


def _evaluate(
    model: E2EAIModel,
    loader: Any,
    source: Any,
    device: torch.device,
    config: Any,
) -> dict[str, Any]:
    score_frame = _collect_scores(model, loader, device)
    score_frame["date"] = _canonical_dates(score_frame["date"])
    weight_frame, diagnostics, targets = _build_weights(score_frame)
    dates = sorted(diagnostics["date"].tolist())
    execution_returns: list[dict[str, float]] = []
    for date, group in score_frame.groupby("date", sort=True):
        execution_returns.append(
            {
                str(asset): float(value)
                for asset, value in zip(group["asset_id"], pd.to_numeric(group["execution_1d_return"], errors="coerce"))
                if np.isfinite(value)
            }
        )
    benchmark = _benchmark_for_dates(source, dates)
    performance, daily = _performance(
        targets,
        execution_returns,
        benchmark,
        [0.0, 5.0, 7.5, 10.0],
        annual_trading_days=config.evaluation.annual_trading_days,
        holding_threshold=config.evaluation.holding_threshold,
    )
    daily.insert(0, "date", dates)
    decile_frame, structure = _structure_features(weight_frame)
    membership_frame, membership = _membership_decomposition(dates, weight_frame, daily)
    diagnostics = diagnostics.merge(decile_frame, on="date", how="left")
    diagnostics = diagnostics.merge(membership_frame, on="date", how="left", suffixes=("", "_membership"))
    score_stats = score_frame["raw_score"].to_numpy(dtype=np.float64)
    rankics: list[float] = []
    for _, group in score_frame.groupby("date", sort=True):
        valid = np.isfinite(group["raw_score"].to_numpy(dtype=np.float64)) & np.isfinite(
            pd.to_numeric(group["execution_1d_return"], errors="coerce").to_numpy(dtype=np.float64)
        )
        if valid.sum() >= 3:
            rankics.append(_rank_corr(group.loc[valid, "raw_score"].to_numpy(dtype=np.float64), pd.to_numeric(group.loc[valid, "execution_1d_return"], errors="coerce").to_numpy(dtype=np.float64)))
    metrics: dict[str, Any] = {
        "validation_execution_rankic": float(np.mean(rankics)) if rankics else 0.0,
        "validation_execution_rankic_std": float(np.std(rankics)) if rankics else 0.0,
        "validation_execution_rankic_count": float(len(rankics)),
        "validation_score_mean": float(np.mean(score_stats)),
        "validation_score_std": float(np.std(score_stats)),
        "validation_score_p01": float(np.quantile(score_stats, 0.01)),
        "validation_score_p50": float(np.quantile(score_stats, 0.50)),
        "validation_score_p99": float(np.quantile(score_stats, 0.99)),
        "validation_effective_n": float(diagnostics["effective_n"].mean()),
        "validation_effective_n_median": float(diagnostics["effective_n"].median()),
        "validation_effective_n_min": float(diagnostics["effective_n"].min()),
        "validation_max_weight_mean": float(diagnostics["max_weight"].mean()),
        "validation_max_weight_p95": float(diagnostics["max_weight"].quantile(0.95)),
        "validation_max_weight": float(diagnostics["max_weight"].max()),
        "validation_hhi": float(diagnostics["hhi"].mean()),
        "validation_normalized_entropy": float(diagnostics["normalized_entropy"].mean()),
        "mean_n_weight_gt_0_5pct": float(diagnostics["n_weight_gt_0_5pct"].mean()),
        "mean_n_weight_gt_1pct": float(diagnostics["n_weight_gt_1pct"].mean()),
        "validation_daily_turnover": float(performance["validation_daily_turnover"]),
        "D10_weight_share": structure["D10_weight_share"],
        "D9_D10_weight_share": structure["D9_D10_weight_share"],
        "D1_weight_share": structure["D1_weight_share"],
        "membership_decomposition": membership,
        "score_weight_structure": structure,
        **performance,
    }
    return {
        "score_frame": score_frame,
        "weight_frame": weight_frame,
        "diagnostics": diagnostics,
        "decile_frame": decile_frame,
        "membership_frame": membership_frame,
        "daily_returns": daily,
        "metrics": metrics,
        "dates": dates,
        "targets": targets,
        "execution_returns": execution_returns,
    }


def _step0_identity(b_result: dict[str, Any], b_output: Path, c_result: dict[str, Any]) -> dict[str, Any]:
    b_weights = pd.read_parquet(b_output / "validation_weights.parquet")
    c_weights = c_result["weight_frame"]
    left = b_weights[["date", "asset", "raw_score", "normalized_score", "weight", "valid_asset"]].copy()
    right = c_weights[["date", "asset", "raw_score", "normalized_score", "weight", "valid_asset"]].copy()
    for frame in (left, right):
        frame["date"] = _canonical_dates(frame["date"])
        frame["asset"] = frame["asset"].astype(str)
    left = left.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    right = right.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    same_axis = left[["date", "asset"]].astype(str).equals(right[["date", "asset"]].astype(str))
    max_raw = float(np.max(np.abs(left["raw_score"].to_numpy() - right["raw_score"].to_numpy()))) if len(left) == len(right) else None
    max_weight = float(np.max(np.abs(left["weight"].to_numpy() - right["weight"].to_numpy()))) if len(left) == len(right) else None
    max_z = float(np.max(np.abs(left["normalized_score"].to_numpy() - right["normalized_score"].to_numpy()))) if len(left) == len(right) else None
    b_daily = pd.read_csv(b_output / "validation_daily_returns.csv")
    c_daily = c_result["daily_returns"]
    daily_columns = ["date", "gross_return", "one_way_turnover", "net_return_0p0bps", "net_return_5p0bps", "net_return_7p5bps", "net_return_10p0bps"]
    for frame in (b_daily, c_daily):
        frame["date"] = _canonical_dates(frame["date"])
    max_daily = float(np.max(np.abs(b_daily[daily_columns[1:]].to_numpy(dtype=float) - c_daily[daily_columns[1:]].to_numpy(dtype=float)))) if len(b_daily) == len(c_daily) else None
    b_metrics = json.loads((b_output / "validation_metrics.json").read_text(encoding="utf-8"))
    checks = {
        "raw_score_max_abs_diff": max_raw,
        "normalized_score_max_abs_diff": max_z,
        "weight_max_abs_diff": max_weight,
        "daily_returns_max_abs_diff": max_daily,
        "step0_7p5bp_sharpe_diff": float(c_result["metrics"]["validation_daily_sharpe_cost_7p5bps"] - b_metrics["validation_daily_sharpe_cost_7p5bps"]),
        "step0_turnover_diff": float(c_result["metrics"]["validation_daily_turnover"] - b_metrics["validation_daily_turnover"]),
        "step0_effective_n_diff": float(c_result["metrics"]["validation_effective_n"] - b_metrics["validation_effective_n"]),
    }
    passed = bool(
        same_axis
        and max_raw is not None and max_raw <= 1e-6
        and max_z is not None and max_z <= 1e-6
        and max_weight is not None and max_weight <= 1e-6
        and max_daily is not None and max_daily <= 1e-6
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "same_date_asset_axis": same_axis,
        "shape_b": list(left.shape),
        "shape_c": list(right.shape),
        "checks": checks,
        "tolerance": 1e-6,
        "step0_metrics_reproduce_b": passed,
        "b_source": str(b_output),
    }


def _save_checkpoint(path: Path, model: E2EAIModel, optimizer: torch.optim.Optimizer, step: int, best_step: int, best_metric: float | None) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": int(step),
            "best_epoch": int(best_step),
            "best_validation_metric": best_metric,
            "training_experiment": "experiment_c",
        },
        path,
    )


def _write_comparison(destination: Path, b_metrics: dict[str, Any], c_metrics: dict[str, Any], factor_metrics: dict[str, Any]) -> None:
    keys = {
        "validation_execution_rankic": "validation_execution_rankic",
        "0bp_annual_return": "validation_daily_return_cost_0p0bps",
        "0bp_sharpe": "validation_daily_sharpe_cost_0p0bps",
        "0bp_mdd": "validation_daily_mdd_cost_0p0bps",
        "7.5bp_annual_return": "validation_daily_return_cost_7p5bps",
        "7.5bp_sharpe": "validation_daily_sharpe_cost_7p5bps",
        "7.5bp_mdd": "validation_daily_mdd_cost_7p5bps",
        "turnover": "validation_daily_turnover",
        "annual_excess_return": "validation_active_annual_excess_return_cost_7p5bps",
        "tracking_error": "validation_tracking_error_cost_7p5bps",
        "information_ratio": "validation_information_ratio_cost_7p5bps",
        "active_max_drawdown": "validation_active_max_drawdown_cost_7p5bps",
        "effective_n": "validation_effective_n",
        "max_weight": "validation_max_weight_mean",
        "hhi": "validation_hhi",
        "normalized_entropy": "validation_normalized_entropy",
        "D10_weight_share": "D10_weight_share",
        "D9_D10_weight_share": "D9_D10_weight_share",
        "n_weight_gt_0.5pct": "mean_n_weight_gt_0_5pct",
        "n_weight_gt_1pct": "mean_n_weight_gt_1pct",
    }
    rows = [{"metric": label, "B": b_metrics.get(key), "C": c_metrics.get(key), "FactorMean": factor_metrics.get(key)} for label, key in keys.items()]
    pd.DataFrame(rows).to_csv(destination / "B_vs_C.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Experiment B vs Experiment C vs Factor Mean",
        "",
        "B is the frozen A-score continuous allocator. C starts from the same A checkpoint and trains only through gross portfolio return with the B allocator unchanged.",
        "",
        "| Metric | B | C | Factor Mean EW |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['B']} | {row['C']} | {row['FactorMean']} |")
    lines.extend(
        [
            "",
            "No delta columns are included by design; the values are shown side-by-side for direct inspection.",
            "C uses no transaction-cost, turnover, cap, TopK, temperature, industry, style, or benchmark loss.",
        ]
    )
    (destination / "B_vs_C.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, *, output: Path | None = None, a_output: Path | None = None, b_output: Path | None = None) -> Path:
    a_config = load_config(config_path)
    if a_config.experiment_line != "daily_strategy" or a_config.model.strategy_head != "simple_mlp":
        raise ValueError("Experiment C requires the accepted daily-core Experiment A config")
    if a_config.model.horizons != [2] or a_config.model.execution_horizon != 2:
        raise ValueError("Experiment C requires execution-only h=2")
    if a_config.data.test_start is not None or a_config.data.test_end is not None:
        raise ValueError("Experiment C must not configure an OOS/test range")
    a_destination = a_output or Path(a_config.logging.output_dir)
    b_destination = b_output or (a_destination.parent / "experiment_b")
    destination = output or (a_destination.parent / "experiment_c")
    checkpoint_a = a_destination / "best.pt"
    if not checkpoint_a.is_file():
        raise FileNotFoundError(f"Missing A checkpoint: {checkpoint_a}")
    if not (b_destination / "validation_weights.parquet").is_file():
        raise FileNotFoundError(f"Missing formal B weights: {b_destination}")
    factor_metrics_path = a_destination.parent / "factor_mean_baseline" / "validation_metrics.json"
    if not factor_metrics_path.is_file():
        raise FileNotFoundError(f"Missing Factor Mean baseline metrics: {factor_metrics_path}")
    destination.mkdir(parents=True, exist_ok=True)
    for child in destination.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_a, checkpoints / "initial_from_a.pt")
    a_sha = _sha256(checkpoint_a)

    seed_everything(a_config.seed)
    source, synthetic = load_configured_data(a_config)
    loaders = build_dataloaders(a_config, source)
    device = resolve_device(a_config.device)
    model = E2EAIModel(a_config.model, a_config.ablation).to(device)
    payload = torch.load(checkpoint_a, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])

    # Step 0 is evaluated before any backward or optimizer operation.
    step0 = _evaluate(model, loaders.validation, source, device, a_config)
    identity = _step0_identity(step0, b_destination, step0)
    write_metrics(destination / "step0_vs_b_identity.json", identity)
    print(
        f"[C] step0 identity={identity['status']} "
        f"rankic={step0['metrics']['validation_execution_rankic']:.6f} "
        f"sharpe_0bp={step0['metrics']['validation_daily_sharpe_cost_0p0bps']:.6f} "
        f"sharpe_7.5bp={step0['metrics']['validation_daily_sharpe_cost_7p5bps']:.6f}",
        flush=True,
    )
    if identity["status"] != "PASS":
        raise RuntimeError(f"C step0 cannot reproduce B; training was not started: {identity}")

    calibration = _calibrate(model, loaders.train, device, max_batches=16)
    write_metrics(destination / "gradient_scale_calibration.json", calibration)
    print(
        f"[C] calibration batches={calibration['batches_used']} "
        f"median_g_score={calibration['median_g_score']:.6g} "
        f"median_g_gross={calibration['median_g_gross']:.6g} "
        f"lambda_portfolio={calibration['lambda_portfolio']:.6g}",
        flush=True,
    )
    c_config = a_config.to_dict()
    c_config["device"] = str(device)
    c_config["model"]["portfolio"]["allocation_mode"] = "long_only_softmax"
    c_config["model"]["portfolio"]["min_selected_stocks"] = 1
    c_config["loss"].update(
        {
            "lambda_score": 1.0,
            "lambda_portfolio": calibration["lambda_portfolio"],
            "lambda_s": 0.0,
            "lambda_f": 0.0,
            "lambda_e": 0.0,
        }
    )
    c_config["training"]["validation_target"] = "validation_daily_sharpe"
    c_config["evaluation"]["cost_scenarios_bps"] = [0.0, 5.0, 7.5, 10.0]
    c_config["experiment_c"] = {
        "initialization": "Experiment A best checkpoint",
        "allocator": "B continuous zscore + masked softmax, temperature=1.0",
        "gross_return_loss": "-mean daily execution return",
        "transaction_cost_loss": "disabled",
        "turnover_loss": "disabled",
        "industry_loss": "disabled",
        "style_loss": "disabled",
        "benchmark_loss": "disabled",
        "topk": False,
        "cap": False,
        "temperature_tuning": False,
        "validation_frequency_steps": 10,
        "early_stopping_patience_evaluations": 20,
        "validation_period": [str(step0["dates"][0]), str(step0["dates"][-1])],
        "no_2022": True,
    }
    (destination / "resolved_config.yaml").write_text(yaml.safe_dump(c_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    optimizer = torch.optim.AdamW(_mlp_parameters(model), lr=a_config.global_optimizer.lr, weight_decay=a_config.global_optimizer.weight_decay)
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_metric = float(step0["metrics"]["validation_daily_sharpe_cost_0p0bps"])
    best_step = 0
    step = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    interval: list[dict[str, float]] = []
    steps_per_epoch = max(1, len(loaders.train))
    total_steps = 0
    stopped_early = False
    component_score, component_gross = calibration["median_g_score"], calibration["median_g_gross"]
    print(
        f"[C] training start epochs={a_config.training.epochs} "
        f"steps_per_epoch={steps_per_epoch} validation_every=10 "
        f"patience_evaluations={a_config.training.early_stopping_patience}",
        flush=True,
    )

    history.append(
        {
            "optimizer_step": 0,
            "epoch_fraction": 0.0,
            "train_score_loss": None,
            "train_gross_loss": None,
            "train_total_loss": None,
            "train_rankic": None,
            "validation_rankic": step0["metrics"]["validation_execution_rankic"],
            "validation_sharpe_0bps": step0["metrics"]["validation_daily_sharpe_cost_0p0bps"],
            "validation_sharpe_7_5bps": step0["metrics"]["validation_daily_sharpe_cost_7p5bps"],
            "validation_annual_return_0bps": step0["metrics"]["validation_daily_return_cost_0p0bps"],
            "validation_annual_return_7_5bps": step0["metrics"]["validation_daily_return_cost_7p5bps"],
            "validation_turnover": step0["metrics"]["validation_daily_turnover"],
            "validation_effective_n": step0["metrics"]["validation_effective_n"],
            "validation_max_weight": step0["metrics"]["validation_max_weight"],
            "validation_hhi": step0["metrics"]["validation_hhi"],
            "validation_normalized_entropy": step0["metrics"]["validation_normalized_entropy"],
            "validation_information_ratio_7_5bps": step0["metrics"]["validation_information_ratio_cost_7p5bps"],
            "D10_weight_share": step0["metrics"]["D10_weight_share"],
            "gradient_norm_total": None,
            "gradient_norm_score_component": component_score,
            "gradient_norm_portfolio_component": component_gross,
            "improved": True,
        }
    )

    # A valid checkpoint container is written at step zero, while the copied
    # initial_from_a.pt remains byte-identical to A's checkpoint.
    _save_checkpoint(checkpoints / "best.pt", model, optimizer, 0, 0, best_metric)
    for epoch in range(a_config.training.epochs):
        for raw_batch in loaders.train:
            batch = _move_batch(dict(raw_batch), device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            scores = _raw_scores(model, batch)
            score_loss, train_ic = _score_loss(scores, batch)
            gross_loss, gross_returns, _, _ = _gross_loss(scores, batch)
            total_loss = a_config.loss.lambda_score * score_loss + calibration["lambda_portfolio"] * gross_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite Experiment C total loss")
            total_loss.backward()
            total_grad = float(torch.nn.utils.clip_grad_norm_(_mlp_parameters(model), a_config.training.gradient_clip_norm, error_if_nonfinite=True))
            optimizer.step()
            step += 1
            total_steps = step
            interval.append(
                {
                    "train_score_loss": float(score_loss.detach().cpu()),
                    "train_gross_loss": float(gross_loss.detach().cpu()),
                    "train_total_loss": float(total_loss.detach().cpu()),
                    "train_rankic": float(np.mean([_rank_corr(scores[row].detach().cpu().numpy()[_valid_execution(batch, scores)[row].detach().cpu().numpy()], batch["execution_1d_return"][row].detach().cpu().numpy()[_valid_execution(batch, scores)[row].detach().cpu().numpy()]) for row in range(scores.shape[0]) if int(_valid_execution(batch, scores)[row].sum()) >= 3])) if scores.shape[0] else 0.0,
                    "gradient_norm_total": total_grad,
                }
            )
            if step % 10 != 0:
                continue
            evaluation = _evaluate(model, loaders.validation, source, device, a_config)
            metric = float(evaluation["metrics"]["validation_daily_sharpe_cost_0p0bps"])
            improved = metric > best_metric + 1e-12
            if improved:
                best_metric = metric
                best_step = step
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            train_means = {key: float(np.mean([row[key] for row in interval])) for key in interval[0]} if interval else {}
            interval = []
            row = {
                "optimizer_step": step,
                "epoch_fraction": float(step / steps_per_epoch),
                **train_means,
                "validation_rankic": evaluation["metrics"]["validation_execution_rankic"],
                "validation_sharpe_0bps": evaluation["metrics"]["validation_daily_sharpe_cost_0p0bps"],
                "validation_sharpe_7_5bps": evaluation["metrics"]["validation_daily_sharpe_cost_7p5bps"],
                "validation_annual_return_0bps": evaluation["metrics"]["validation_daily_return_cost_0p0bps"],
                "validation_annual_return_7_5bps": evaluation["metrics"]["validation_daily_return_cost_7p5bps"],
                "validation_turnover": evaluation["metrics"]["validation_daily_turnover"],
                "validation_effective_n": evaluation["metrics"]["validation_effective_n"],
                "validation_max_weight": evaluation["metrics"]["validation_max_weight"],
                "validation_hhi": evaluation["metrics"]["validation_hhi"],
                "validation_normalized_entropy": evaluation["metrics"]["validation_normalized_entropy"],
                "validation_information_ratio_7_5bps": evaluation["metrics"]["validation_information_ratio_cost_7p5bps"],
                "D10_weight_share": evaluation["metrics"]["D10_weight_share"],
                "gradient_norm_score_component": component_score,
                "gradient_norm_portfolio_component": component_gross,
                "improved": improved,
            }
            history.append(row)
            print(
                f"[C] step={step} epoch={row['epoch_fraction']:.3f} "
                f"train_total={row.get('train_total_loss', float('nan')):.6g} "
                f"val_sharpe_0bp={row['validation_sharpe_0bps']:.6f} "
                f"val_sharpe_7.5bp={row['validation_sharpe_7_5bps']:.6f} "
                f"effN={row['validation_effective_n']:.2f} "
                f"maxW={row['validation_max_weight']:.4f} "
                f"improved={improved} no_improve={no_improve}",
                flush=True,
            )
            if improved:
                _save_checkpoint(checkpoints / "best.pt", model, optimizer, step, best_step, best_metric)
            if no_improve >= a_config.training.early_stopping_patience:
                stopped_early = True
                print(f"[C] early stop at step={step} after {no_improve} non-improving validations", flush=True)
                break
        if stopped_early:
            break
    _save_checkpoint(checkpoints / "last.pt", model, optimizer, total_steps, best_step, best_metric)
    model.load_state_dict(best_state)
    final = _evaluate(model, loaders.validation, source, device, a_config)
    final_metrics = dict(final["metrics"])
    best_history = [row for row in history if int(row.get("optimizer_step", -1)) == best_step]
    train_rankic_best = float(best_history[0].get("train_rankic")) if best_history and best_history[0].get("train_rankic") is not None else float("nan")
    final_metrics.update(
        {
            "experiment": "experiment_c",
            "training_performed": True,
            "optimizer_steps_total": total_steps,
            "best_optimizer_step": best_step,
            "best_epoch_fraction": float(best_step / steps_per_epoch),
            "early_stopped": stopped_early,
            "train_rankic_at_best": train_rankic_best,
            "validation_rankic_at_best": final_metrics["validation_execution_rankic"],
            "ic_gap_train_minus_validation": train_rankic_best - final_metrics["validation_execution_rankic"] if np.isfinite(train_rankic_best) else None,
            "lambda_score": 1.0,
            "lambda_portfolio": calibration["lambda_portfolio"],
            "gradient_calibration_batches": calibration["batches_used"],
            "no_2022_data": True,
            "no_transaction_cost_loss": True,
            "no_turnover_loss": True,
            "no_topk_or_cap": True,
            "initial_checkpoint_sha256": a_sha,
            "initial_from_a_sha256": _sha256(checkpoints / "initial_from_a.pt"),
            "best_checkpoint_sha256": _sha256(checkpoints / "best.pt"),
            "last_checkpoint_sha256": _sha256(checkpoints / "last.pt"),
            "validation_date_range": [final["dates"][0], final["dates"][-1]],
            "validation_date_count": len(final["dates"]),
            "synthetic": synthetic,
            "b_step0_metrics": step0["metrics"],
        }
    )
    final["score_frame"].to_parquet(destination / "validation_scores.parquet", index=False)
    final["weight_frame"].to_parquet(destination / "validation_weights.parquet", index=False)
    final["diagnostics"].to_csv(destination / "validation_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    final["daily_returns"].to_csv(destination / "validation_daily_returns.csv", index=False, encoding="utf-8-sig")
    final["decile_frame"].to_csv(destination / "validation_score_decile_weights.csv", index=False, encoding="utf-8-sig")
    final["membership_frame"].to_csv(destination / "turnover_membership_decomposition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(destination / "training_steps.csv", index=False, encoding="utf-8-sig")
    write_metrics(destination / "validation_metrics.json", final_metrics)
    factor_metrics = json.loads(factor_metrics_path.read_text(encoding="utf-8"))
    b_metrics = json.loads((b_destination / "validation_metrics.json").read_text(encoding="utf-8"))
    b_weights = pd.read_parquet(b_destination / "validation_weights.parquet")
    _, b_structure = _structure_features(b_weights)
    b_metrics = {**b_metrics, **b_structure, "validation_max_weight_mean": b_metrics.get("validation_max_weight_mean", b_metrics.get("validation_max_weight")), "mean_n_weight_gt_0_5pct": b_metrics.get("mean_n_weight_gt_0_5pct"), "mean_n_weight_gt_1pct": b_metrics.get("mean_n_weight_gt_1pct")}
    _write_comparison(destination, b_metrics, final_metrics, factor_metrics)
    manifest = {
        "experiment": "experiment_c",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment_a": str(a_destination),
        "source_experiment_b": str(b_destination),
        "output_dir": str(destination),
        "initial_from_a_sha256": _sha256(checkpoints / "initial_from_a.pt"),
        "optimizer_steps_total": total_steps,
        "best_optimizer_step": best_step,
        "early_stopped": stopped_early,
        "no_experiment_d": True,
        "no_2022": True,
        "files": sorted({path.name for path in destination.iterdir()} | {"run_manifest.json"}),
    }
    write_metrics(destination / "run_manifest.json", manifest)
    print(
        f"[C] completed steps={total_steps} best_step={best_step} "
        f"best_sharpe_0bp={best_metric:.6f} early_stopped={stopped_early} "
        f"output={destination}",
        flush=True,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--a-output", type=Path)
    parser.add_argument("--b-output", type=Path)
    args = parser.parse_args()
    print(run(args.config, output=args.output, a_output=args.a_output, b_output=args.b_output))


if __name__ == "__main__":
    main()
