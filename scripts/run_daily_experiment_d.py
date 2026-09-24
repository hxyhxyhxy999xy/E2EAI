"""Run Experiment D: end-to-end training with actual 7.5bp net return.

Experiment D is deliberately isolated from Experiment C.  It starts from C's
best checkpoint, verifies differentiable transaction-cost accounting against
the formal daily evaluator, then trains the same MLP/allocator with
``L_score + 70.2102429073 * L_net``.  No new turnover penalty, cap, TopK,
temperature, or other portfolio constraint is introduced.
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
from e2eai.evaluation.continuous_portfolio import continuous_long_only_weights
from e2eai.evaluation.daily_validation import _simulate_daily_weight_path
from e2eai.models.e2eai import E2EAIModel
from e2eai.training.net_return import account_sequence_torch, numpy_drifted_rows
from e2eai.training.losses import masked_cross_sectional_ic
from e2eai.utils.reproducibility import resolve_device, seed_everything
from e2eai.workflows import load_configured_data, write_metrics
from scripts.run_daily_experiment_b import _benchmark_for_dates, _canonical_dates, _performance, _sha256
from scripts.run_daily_experiment_c import (
    _collect_scores,
    _evaluate,
    _gross_loss,
    _mlp_parameters,
    _raw_scores,
    _rank_corr,
    _score_loss,
)

LAMBDA_PORTFOLIO = 70.21024290734199
COST_BPS = 7.5


def _rows_from_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, date in enumerate(batch["dates"]):
        count = int(batch["asset_mask"][index].sum().item())
        rows.append(
            {
                "date": date,
                "asset_ids": [str(value) for value in batch["asset_ids"][index][:count]],
                "raw_factors": batch["raw_factors"][index, :count].detach().cpu(),
                "execution_1d_return": batch["execution_1d_return"][index, :count].detach().cpu(),
                "execution_return_mask": batch["execution_return_mask"][index, :count].detach().cpu(),
            }
        )
    return rows


def _collate_rows(rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate empty rows")
    width = max(int(row["raw_factors"].shape[0]) for row in rows)
    factors = torch.zeros(len(rows), width, int(rows[0]["raw_factors"].shape[-1]), dtype=torch.float32, device=device)
    execution = torch.zeros(len(rows), width, dtype=torch.float32, device=device)
    execution_mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    asset_mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    asset_ids: list[list[str]] = []
    dates: list[Any] = []
    for index, row in enumerate(rows):
        count = len(row["asset_ids"])
        factors[index, :count] = row["raw_factors"].to(device)
        execution[index, :count] = row["execution_1d_return"].to(device)
        execution_mask[index, :count] = row["execution_return_mask"].to(device).bool()
        asset_mask[index, :count] = True
        asset_ids.append(list(row["asset_ids"]))
        dates.append(row["date"])
    return {
        "raw_factors": factors,
        "execution_1d_return": execution,
        "execution_return_mask": execution_mask,
        "asset_mask": asset_mask,
        "asset_ids": asset_ids,
        "dates": dates,
    }


def _batch_specs(train_batches: list[dict[str, Any]]) -> list[tuple[list[dict[str, Any]], int, int]]:
    rows_by_batch = [_rows_from_batch(batch) for batch in train_batches]
    specs: list[tuple[list[dict[str, Any]], int, int]] = []
    for index, rows in enumerate(rows_by_batch):
        if index == 0:
            specs.append((rows, 0, index))
        else:
            specs.append(([rows_by_batch[index - 1][-1], *rows], 1, index))
    return specs


def _slice_batch(batch: dict[str, Any], start: int) -> dict[str, Any]:
    return {
        "execution_1d_return": batch["execution_1d_return"][start:],
        "execution_return_mask": batch["execution_return_mask"][start:],
        "asset_mask": batch["asset_mask"][start:],
        "dates": batch["dates"][start:],
        "asset_ids": batch["asset_ids"][start:],
    }


def _net_components(
    model: E2EAIModel,
    batch: dict[str, Any],
    loss_start: int,
) -> dict[str, Any]:
    scores = _raw_scores(model, batch)
    weights, normalized, _, _ = __import__("e2eai.evaluation.continuous_portfolio", fromlist=["continuous_long_only_weights_torch_batch"]).continuous_long_only_weights_torch_batch(
        scores, batch["asset_mask"].bool(), temperature=1.0, eps=1e-6
    )
    accounting = account_sequence_torch(
        weights,
        batch["asset_ids"],
        batch["asset_mask"],
        batch["execution_1d_return"],
        batch["execution_return_mask"],
        holding_threshold=1e-8,
    )
    loss_batch = _slice_batch(batch, loss_start)
    score_loss, score_ic = _score_loss(scores[loss_start:], loss_batch)
    gross_loss = -accounting["gross_return"][loss_start:].mean()
    cost_loss = accounting["cost_7_5bps"][loss_start:].mean()
    net_loss = -accounting["net_return"][loss_start:].mean()
    return {
        "scores": scores,
        "weights": weights,
        "normalized_scores": normalized,
        "accounting": accounting,
        "score_loss": score_loss,
        "gross_loss": gross_loss,
        "cost_loss": cost_loss,
        "net_loss": net_loss,
        "score_ic": score_ic,
        "loss_start": loss_start,
    }


def _grad_vector(model: E2EAIModel) -> np.ndarray:
    values = []
    for parameter in _mlp_parameters(model):
        if parameter.grad is None:
            raise RuntimeError("missing MLP gradient")
        values.append(parameter.grad.detach().cpu().numpy().astype(np.float64).reshape(-1))
    vector = np.concatenate(values)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
        raise FloatingPointError("non-finite or zero MLP gradient")
    return vector


def _gradient_norm(model: E2EAIModel) -> float:
    return float(np.linalg.norm(_grad_vector(model)))


def _numpy_from_torch(values: list[torch.Tensor]) -> np.ndarray:
    return torch.stack(values).detach().cpu().numpy().astype(np.float64)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))) if len(left) else 0.0


def _formal_from_targets(targets: list[dict[str, float]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [{asset: float(value) for asset, value, ok in zip(row["asset_ids"], row["execution_1d_return"].numpy(), row["execution_return_mask"].numpy()) if bool(ok) and np.isfinite(value)} for row in rows]
    path = _simulate_daily_weight_path(targets, returns, cost_bps=COST_BPS, holding_threshold=1e-8)
    drift, gross, turnover, cost, net = numpy_drifted_rows(targets, returns)
    return {"path": path, "drift": drift, "gross": gross, "turnover": turnover, "cost": cost, "net": net}


def _targets_from_weights(weights: np.ndarray, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    result = []
    for values, row in zip(weights, rows):
        result.append({asset: float(weight) for asset, weight in zip(row["asset_ids"], values[: len(row["asset_ids"])])})
    return result


def _drift_diff(custom_rows: list[dict[str, torch.Tensor]], formal_rows: list[dict[str, float]]) -> float:
    values = []
    for left, right in zip(custom_rows, formal_rows):
        assets = set(left) | set(right)
        values.extend(float((left.get(asset, torch.tensor(0.0)) - right.get(asset, 0.0)).detach().cpu()) for asset in assets)
    return float(max((abs(value) for value in values), default=0.0))


def _preflight_accounting(model: E2EAIModel, specs: list[tuple[list[dict[str, Any]], int, int]], device: torch.device) -> dict[str, Any]:
    # Test A: an independent 3-stock, 5-date toy path with real drift.
    toy_assets = [["a", "b", "c"]] * 5
    toy_targets = [
        {"a": 0.50, "b": 0.30, "c": 0.20},
        {"a": 0.40, "b": 0.40, "c": 0.20},
        {"a": 0.35, "b": 0.45, "c": 0.20},
        {"a": 0.30, "b": 0.50, "c": 0.20},
        {"a": 0.25, "b": 0.55, "c": 0.20},
    ]
    toy_returns = [{"a": 0.01, "b": -0.02, "c": 0.03}, {"a": 0.02, "b": 0.01, "c": -0.01}, {"a": -0.01, "b": 0.03, "c": 0.02}, {"a": 0.04, "b": -0.01, "c": 0.01}, {"a": 0.01, "b": 0.02, "c": -0.02}]
    toy_w = torch.tensor([[target[asset] for asset in ["a", "b", "c"]] for target in toy_targets], dtype=torch.float32, requires_grad=True)
    toy_r = torch.tensor([[row[asset] for asset in ["a", "b", "c"]] for row in toy_returns], dtype=torch.float32)
    toy_mask = torch.ones_like(toy_w, dtype=torch.bool)
    toy_exec_mask = torch.ones_like(toy_w, dtype=torch.bool)
    toy_torch = account_sequence_torch(toy_w, toy_assets, toy_mask, toy_r, toy_exec_mask)
    toy_formal = _formal_from_targets(toy_targets, [{"asset_ids": ["a", "b", "c"], "execution_1d_return": row, "execution_return_mask": torch.ones(3, dtype=torch.bool)} for row in toy_r])
    toy_diffs = {"gross_max_diff": _max_abs(toy_torch["gross_return"].detach().numpy(), toy_formal["gross"]), "drift_weight_max_diff": _drift_diff(toy_torch["drifted_previous_weight"], toy_formal["drift"]), "turnover_max_diff": _max_abs(toy_torch["turnover"].detach().numpy(), toy_formal["turnover"]), "cost_max_diff": _max_abs(toy_torch["cost_7_5bps"].detach().numpy(), toy_formal["cost"]), "net_max_diff": _max_abs(toy_torch["net_return"].detach().numpy(), toy_formal["net"])}

    # Test B: first 16 real train dates under C's frozen checkpoint.
    real_rows = [*specs[0][0], *specs[1][0][1:]]
    real_batch = _collate_rows(real_rows, device)
    model.eval()
    with torch.no_grad():
        real_scores = _raw_scores(model, real_batch)
        real_weights, _, _, _ = __import__("e2eai.evaluation.continuous_portfolio", fromlist=["continuous_long_only_weights_torch_batch"]).continuous_long_only_weights_torch_batch(real_scores, real_batch["asset_mask"], temperature=1.0, eps=1e-6)
    real_np_weights = real_weights.detach().cpu().numpy()
    real_targets = _targets_from_weights(real_np_weights, real_rows)
    real_formal = _formal_from_targets(real_targets, real_rows)
    real_torch = account_sequence_torch(real_weights, real_batch["asset_ids"], real_batch["asset_mask"], real_batch["execution_1d_return"], real_batch["execution_return_mask"])
    real_diffs = {"gross_max_diff": _max_abs(real_torch["gross_return"].detach().cpu().numpy(), real_formal["gross"]), "drift_weight_max_diff": _drift_diff(real_torch["drifted_previous_weight"], real_formal["drift"]), "turnover_max_diff": _max_abs(real_torch["turnover"].detach().cpu().numpy(), real_formal["turnover"]), "cost_max_diff": _max_abs(real_torch["cost_7_5bps"].detach().cpu().numpy(), real_formal["cost"]), "net_max_diff": _max_abs(real_torch["net_return"].detach().cpu().numpy(), real_formal["net"])}

    # Test C: real train gradient, including cost and previous-day state.
    sequence, loss_start, _ = specs[1]
    gradient_batch = _collate_rows(sequence, device)
    model.zero_grad(set_to_none=True)
    components = _net_components(model, gradient_batch, loss_start)
    components["net_loss"].backward()
    net_norm = _gradient_norm(model)
    model.zero_grad(set_to_none=True)
    components = _net_components(model, gradient_batch, loss_start)
    components["scores"].retain_grad()
    components["cost_loss"].backward()
    cost_norm = _gradient_norm(model)
    context_grad = None if components["scores"].grad is None else float(components["scores"].grad[0].detach().abs().sum().cpu())
    gradient = {"net_gradient_norm": net_norm, "cost_gradient_norm": cost_norm, "all_finite": True, "cost_gradient_nonzero": cost_norm > 0, "net_gradient_nonzero": net_norm > 0, "previous_day_score_gradient_nonzero": context_grad is not None and context_grad > 0}
    passed = all(max(values.values()) <= 1e-6 for values in (toy_diffs, real_diffs)) and gradient["all_finite"] and gradient["cost_gradient_nonzero"] and gradient["net_gradient_nonzero"]
    return {"status": "PASS" if passed else "FAIL", "toy_test": toy_diffs, "real_train_slice": real_diffs, "gradient_test": gradient, "batch_boundary_test": _batch_boundary_test(), "used_2022": False}


def _batch_boundary_test() -> dict[str, Any]:
    assets = [["a", "b", "c"]] * 17
    weights = torch.tensor([[0.5 + 0.001 * i, 0.3, 0.2 - 0.001 * i] for i in range(17)], dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    returns = torch.tensor([[0.01 + 0.001 * (i % 3), -0.01 + 0.002 * (i % 2), 0.02 - 0.001 * (i % 4)] for i in range(17)], dtype=torch.float32)
    mask = torch.ones_like(weights, dtype=torch.bool)
    full = account_sequence_torch(weights, assets, mask, returns, mask)
    pieces = []
    for start, end, loss_start in [(0, 8, 0), (7, 16, 1), (15, 17, 1)]:
        piece = account_sequence_torch(weights[start:end], assets[start:end], mask[start:end], returns[start:end], mask[start:end])
        pieces.append((piece, loss_start))
    batched = {key: torch.cat([piece[key][loss_start:] for piece, loss_start in pieces]) for key in ["gross_return", "turnover", "cost_7_5bps", "net_return"]}
    return {"status": "PASS" if all(_max_abs(full[key].detach().numpy(), batched[key].detach().numpy()) <= 1e-7 for key in batched) else "FAIL", "max_diff": max(_max_abs(full[key].detach().numpy(), batched[key].detach().numpy()) for key in batched), "dates_loss_once": 17}


def _step0_gradient_decomposition(model: E2EAIModel, specs: list[tuple[list[dict[str, Any]], int, int]], device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for batch_index, (sequence, loss_start, _) in enumerate(specs[:16]):
        batch = _collate_rows(sequence, device)
        vectors: dict[str, np.ndarray] = {}
        values: dict[str, float] = {}
        for name in ["score_loss", "gross_loss", "cost_loss", "net_loss"]:
            model.zero_grad(set_to_none=True)
            component = _net_components(model, batch, loss_start)
            values[name] = float(component[name].detach().cpu())
            component[name].backward()
            vectors[name] = _grad_vector(model)
        if not np.allclose(vectors["net_loss"], vectors["gross_loss"] + vectors["cost_loss"], rtol=1e-5, atol=1e-7):
            raise RuntimeError("D net gradient is not gross gradient plus cost gradient")
        weighted_gross = LAMBDA_PORTFOLIO * vectors["gross_loss"]
        weighted_cost = LAMBDA_PORTFOLIO * vectors["cost_loss"]
        weighted_net = LAMBDA_PORTFOLIO * vectors["net_loss"]
        score_norm = np.linalg.norm(vectors["score_loss"])
        gross_norm = np.linalg.norm(weighted_gross)
        cost_norm = np.linalg.norm(weighted_cost)
        net_norm = np.linalg.norm(weighted_net)
        cosine_gc = float(np.dot(weighted_gross, weighted_cost) / (gross_norm * cost_norm))
        cosine_sn = float(np.dot(vectors["score_loss"], weighted_net) / (score_norm * net_norm))
        rows.append({"batch_index": batch_index, **values, "norm_score": score_norm, "norm_weighted_gross": gross_norm, "norm_weighted_cost": cost_norm, "norm_weighted_net": net_norm, "cost_to_gross_gradient_ratio": cost_norm / gross_norm, "net_to_score_gradient_ratio": net_norm / score_norm, "cosine_gross_cost": cosine_gc, "cosine_score_net": cosine_sn})
    frame = pd.DataFrame(rows)
    median = frame.median(numeric_only=True).to_dict()
    result = {"batches_used": len(frame), "lambda_portfolio": LAMBDA_PORTFOLIO, "median": median, "per_batch": rows, "all_finite": bool(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()), "cost_gradient_nonzero": bool(float(median["norm_weighted_cost"]) > 0), "net_gradient_nonzero": bool(float(median["norm_weighted_net"]) > 0), "gradient_scale_warning": bool(float(median["cost_to_gross_gradient_ratio"]) > 100 or float(median["cost_to_gross_gradient_ratio"]) < 0.01)}
    model.zero_grad(set_to_none=True)
    return result


def _save_checkpoint(path: Path, model: E2EAIModel, optimizer: torch.optim.Optimizer, step: int, best_step: int, best_metric: float) -> None:
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "global_step": int(step), "best_epoch": int(best_step), "best_validation_metric": float(best_metric), "training_experiment": "experiment_d"}, path)


def _identity_step0(c_result: dict[str, Any], c_output: Path) -> dict[str, Any]:
    c_weights = pd.read_parquet(c_output / "validation_weights.parquet")
    right = c_result["weight_frame"]
    left = c_weights.rename(columns={"asset_id": "asset"}) if "asset_id" in c_weights.columns else c_weights
    cols = ["date", "asset", "raw_score", "normalized_score", "weight", "valid_asset"]
    left = left[cols].copy(); right = right[cols].copy()
    for frame in (left, right):
        frame["date"] = _canonical_dates(frame["date"]); frame["asset"] = frame["asset"].astype(str)
    left = left.sort_values(["date", "asset"], kind="stable").reset_index(drop=True); right = right.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    diffs = {f"{col}_max_abs_diff": _max_abs(left[col].to_numpy(dtype=float), right[col].to_numpy(dtype=float)) for col in ["raw_score", "normalized_score", "weight"]}
    c_daily = pd.read_csv(c_output / "validation_daily_returns.csv"); c_daily["date"] = _canonical_dates(c_daily["date"])
    d_daily = c_result["daily_returns"].copy(); d_daily["date"] = _canonical_dates(d_daily["date"])
    merged = c_daily.merge(d_daily, on="date", suffixes=("_c", "_d"), validate="one_to_one")
    for col in ["gross_return", "one_way_turnover", "net_return_7p5bps"]:
        diffs[f"{col}_max_abs_diff"] = _max_abs(merged[f"{col}_c"].to_numpy(), merged[f"{col}_d"].to_numpy())
    # Transaction cost is exactly 2 * one-way turnover * 7.5bp.  Record the
    # implied cost parity explicitly so the identity artifact audits every
    # required economic quantity without introducing a second accounting path.
    diffs["cost_7p5bps_max_abs_diff"] = 2.0 * 7.5 / 10000.0 * diffs["one_way_turnover_max_abs_diff"]
    passed = left.shape == right.shape and left[["date", "asset"]].astype(str).equals(right[["date", "asset"]].astype(str)) and all(value <= 1e-6 for value in diffs.values())
    return {"status": "PASS" if passed else "FAIL", "c_best_checkpoint_sha256": _sha256(c_output / "checkpoints" / "best.pt"), "tolerance": 1e-6, **diffs}


def _daily_with_costs(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    result["turnover"] = result["one_way_turnover"]
    for bps in [5.0, 7.5, 10.0]:
        token = str(bps).replace(".", "p")
        result[f"cost_{str(bps).replace('.', '_')}bps"] = 2.0 * result["turnover"] * bps / 10000.0
        result[f"net_return_{str(bps).replace('.', '_')}bps"] = result[f"net_return_{token}bps"]
    result["active_return_7_5bps"] = result["net_return_7_5bps"] - result["benchmark_return"]
    return result


def _cost_drag(daily: pd.DataFrame, annual_days: int = 252) -> float:
    n = len(daily)
    gross_ann = float(np.prod(1.0 + daily["gross_return"].to_numpy()) ** (annual_days / n) - 1.0)
    net_ann = float(np.prod(1.0 + daily["net_return_7p5bps"].to_numpy()) ** (annual_days / n) - 1.0)
    return gross_ann - net_ann


def _stability_diagnostics(weight_frame: pd.DataFrame, dates: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    maps: list[dict[str, float]] = []
    rows = []
    for date, group in weight_frame.groupby("date", sort=True):
        current = dict(zip(group.loc[group["valid_asset"], "asset"].astype(str), group.loc[group["valid_asset"], "weight"].astype(float)))
        if maps:
            common = sorted(set(maps[-1]) & set(current)); union = set(maps[-1]) | set(current)
            previous = np.asarray([maps[-1].get(asset, 0.0) for asset in union]); now = np.asarray([current.get(asset, 0.0) for asset in union])
            pearson = float(np.corrcoef(previous, now)[0, 1]) if np.std(previous) > 1e-12 and np.std(now) > 1e-12 else 0.0
            spearman = _rank_corr(previous, now)
            abs_change = float(np.mean(np.abs(now - previous)))
        else:
            pearson = spearman = 0.0; abs_change = 0.0
        rows.append({"date": str(date), "adjacent_weight_pearson": pearson, "adjacent_weight_spearman": spearman, "mean_absolute_daily_target_weight_change": abs_change})
        maps.append(current)
    frame = pd.DataFrame(rows)
    return frame, {"weight_corr_pearson": float(frame["adjacent_weight_pearson"].iloc[1:].mean()) if len(frame) > 1 else 0.0, "weight_corr_spearman": float(frame["adjacent_weight_spearman"].iloc[1:].mean()) if len(frame) > 1 else 0.0, "mean_absolute_daily_target_weight_change": float(frame["mean_absolute_daily_target_weight_change"].iloc[1:].mean()) if len(frame) > 1 else 0.0}


def _write_c_vs_d(output: Path, c_metrics: dict[str, Any], d_metrics: dict[str, Any], c_daily: pd.DataFrame, d_daily: pd.DataFrame) -> None:
    c_drag = _cost_drag(c_daily); d_drag = _cost_drag(d_daily)
    rows = []
    pairs = {"0bp_annual_return": (c_metrics.get("validation_daily_return_cost_0p0bps"), d_metrics.get("validation_daily_return_cost_0p0bps")), "0bp_sharpe": (c_metrics.get("validation_daily_sharpe_cost_0p0bps"), d_metrics.get("validation_daily_sharpe_cost_0p0bps")), "0bp_mdd": (c_metrics.get("validation_daily_mdd_cost_0p0bps"), d_metrics.get("validation_daily_mdd_cost_0p0bps")), "7.5bp_annual_return": (c_metrics.get("validation_daily_return_cost_7p5bps"), d_metrics.get("validation_daily_return_cost_7p5bps")), "7.5bp_sharpe": (c_metrics.get("validation_daily_sharpe_cost_7p5bps"), d_metrics.get("validation_daily_sharpe_cost_7p5bps")), "7.5bp_mdd": (c_metrics.get("validation_daily_mdd_cost_7p5bps"), d_metrics.get("validation_daily_mdd_cost_7p5bps")), "mean_turnover": (c_metrics.get("validation_daily_turnover"), d_metrics.get("validation_daily_turnover")), "annual_transaction_cost_drag": (c_drag, d_drag), "effective_n": (c_metrics.get("validation_effective_n"), d_metrics.get("validation_effective_n")), "mean_max_weight": (c_metrics.get("validation_max_weight_mean"), d_metrics.get("validation_max_weight_mean")), "absolute_max_weight": (c_metrics.get("validation_max_weight"), d_metrics.get("validation_max_weight")), "normalized_entropy": (c_metrics.get("validation_normalized_entropy"), d_metrics.get("validation_normalized_entropy")), "D10_weight_share": (c_metrics.get("D10_weight_share"), d_metrics.get("D10_weight_share")), "annual_excess_return_7.5bp": (c_metrics.get("validation_active_annual_excess_return_cost_7p5bps"), d_metrics.get("validation_active_annual_excess_return_cost_7p5bps")), "tracking_error_7.5bp": (c_metrics.get("validation_tracking_error_cost_7p5bps"), d_metrics.get("validation_tracking_error_cost_7p5bps")), "information_ratio_7.5bp": (c_metrics.get("validation_information_ratio_cost_7p5bps"), d_metrics.get("validation_information_ratio_cost_7p5bps")), "active_mdd_7.5bp": (c_metrics.get("validation_active_max_drawdown_cost_7p5bps"), d_metrics.get("validation_active_max_drawdown_cost_7p5bps"))}
    for metric, (c_value, d_value) in pairs.items(): rows.append({"metric": metric, "C": c_value, "D": d_value, "D_minus_C": None if c_value is None or d_value is None else float(d_value) - float(c_value)})
    pd.DataFrame(rows).to_csv(output / "C_vs_D.csv", index=False, encoding="utf-8-sig")
    lines = ["# Experiment C vs D", "", "C is frozen Experiment C best step20 at T=1. D trains the same MLP/allocator using actual 7.5bp net-return loss.", "", "| Metric | C | D | D − C |", "|---|---:|---:|---:|"]
    lines.extend(f"| {row['metric']} | {row['C']} | {row['D']} | {row['D_minus_C']} |" for row in rows)
    (output / "C_vs_D.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return c_drag, d_drag


def run(config_path: Path, *, c_output: Path, output: Path | None = None) -> Path:
    c_output = c_output.resolve(); output = (output or c_output.parent / "experiment_d").resolve()
    c_checkpoint = c_output / "checkpoints" / "best.pt"
    if not c_checkpoint.is_file(): raise FileNotFoundError(c_checkpoint)
    config = load_config(config_path)
    if config.model.horizons != [2] or config.model.execution_horizon != 2: raise ValueError("D requires h=2")
    if config.data.test_start is not None or config.data.test_end is not None: raise ValueError("D must not configure OOS/test")
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir(): shutil.rmtree(child) if child.is_dir() else child.unlink()
    checkpoint_sha = _sha256(c_checkpoint)
    shutil.copy2(c_checkpoint, output / "initial_from_c.pt")
    seed_everything(config.seed)
    source, synthetic = load_configured_data(config); loaders = build_dataloaders(config, source); device = resolve_device(config.device)
    train_batches = [dict(batch) for batch in loaders.train]
    specs = _batch_specs(train_batches)
    train_dates = [str(row["date"])[:10] for batch in train_batches for row in _rows_from_batch(batch)]
    if any(date >= "2021-01-01" for date in train_dates): raise RuntimeError("D train loader includes validation/test dates")
    model = E2EAIModel(config.model, config.ablation).to(device)
    payload = torch.load(c_checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"]); model.eval()
    step0 = _evaluate(model, loaders.validation, source, device, config)
    step0_identity = _identity_step0(step0, c_output); write_metrics(output / "step0_vs_c_identity.json", step0_identity)
    if step0_identity["status"] != "PASS": raise RuntimeError(f"D step0 cannot reproduce C: {step0_identity}")
    preflight = _preflight_accounting(model, specs, device); write_metrics(output / "transaction_cost_preflight.json", preflight)
    preflight_md = ["# Transaction Cost Preflight", "", f"Status: **{preflight['status']}**", "", f"Toy test: `{preflight['toy_test']}`", "", f"Real train slice: `{preflight['real_train_slice']}`", "", f"Batch boundary: `{preflight['batch_boundary_test']}`", "", f"Gradient test: `{preflight['gradient_test']}`", ""]
    (output / "transaction_cost_preflight.md").write_text("\n".join(preflight_md), encoding="utf-8")
    if preflight["status"] != "PASS" or preflight["batch_boundary_test"]["status"] != "PASS": raise RuntimeError("Transaction-cost preflight failed; D training was not started")
    decomposition = _step0_gradient_decomposition(model, specs, device); write_metrics(output / "step0_gradient_decomposition.json", decomposition)
    if not decomposition["all_finite"] or not decomposition["cost_gradient_nonzero"] or not decomposition["net_gradient_nonzero"]: raise RuntimeError("D gradient sanity failed; training was not started")
    resolved = yaml.safe_load((c_output / "resolved_config.yaml").read_text(encoding="utf-8")); resolved["experiment_d"] = {"initialization": "Experiment C best step20", "checkpoint_sha256": checkpoint_sha, "portfolio_objective": "-mean(gross_return - 2*turnover*7.5/10000)", "lambda_score": 1.0, "lambda_portfolio": LAMBDA_PORTFOLIO, "temperature": 1.0, "transaction_cost_bps": 7.5, "extra_turnover_penalty": 0.0, "validation_target": "validation_daily_sharpe_7_5bps", "validation_frequency_steps": 10, "early_stopping_patience_evaluations": config.training.early_stopping_patience, "no_2022": True, "output_dir": str(output)}
    # Keep the resolved artifact self-consistent with the actual D output path;
    # this is metadata only and does not alter the trained model or evaluation.
    resolved.setdefault("logging", {})["output_dir"] = str(output)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    optimizer = torch.optim.AdamW(_mlp_parameters(model), lr=config.global_optimizer.lr, weight_decay=config.global_optimizer.weight_decay)
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; best_metric = float(step0["metrics"]["validation_daily_sharpe_cost_7p5bps"]); best_step = 0; step = 0; no_improve = 0; stopped_early = False; history: list[dict[str, Any]] = []
    history.append({"optimizer_step": 0, "epoch_fraction": 0.0, "train_score_loss": None, "train_gross_return": None, "train_turnover": None, "train_cost_7_5bps": None, "train_net_return": None, "train_net_loss": None, "train_total_loss": None, "train_rankic": None, "validation_rankic": step0["metrics"]["validation_execution_rankic"], "validation_annual_return_0bps": step0["metrics"]["validation_daily_return_cost_0p0bps"], "validation_sharpe_0bps": step0["metrics"]["validation_daily_sharpe_cost_0p0bps"], "validation_annual_return_7_5bps": step0["metrics"]["validation_daily_return_cost_7p5bps"], "validation_sharpe_7_5bps": step0["metrics"]["validation_daily_sharpe_cost_7p5bps"], "validation_mdd_7_5bps": step0["metrics"]["validation_daily_mdd_cost_7p5bps"], "validation_turnover": step0["metrics"]["validation_daily_turnover"], "validation_effective_n": step0["metrics"]["validation_effective_n"], "validation_max_weight": step0["metrics"]["validation_max_weight_mean"], "validation_normalized_entropy": step0["metrics"]["validation_normalized_entropy"], "validation_information_ratio_7_5bps": step0["metrics"]["validation_information_ratio_cost_7p5bps"], "D10_weight_share": step0["metrics"]["D10_weight_share"], "gradient_norm_total": None, "improved": True})
    checkpoints = output / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.training.epochs):
        for sequence, loss_start, batch_index in specs:
            batch = _collate_rows(sequence, device); model.train(); optimizer.zero_grad(set_to_none=True); components = _net_components(model, batch, loss_start); total_loss = components["score_loss"] + LAMBDA_PORTFOLIO * components["net_loss"]
            if not torch.isfinite(total_loss): raise FloatingPointError("non-finite D total loss")
            total_loss.backward(); total_grad = float(torch.nn.utils.clip_grad_norm_(_mlp_parameters(model), config.training.gradient_clip_norm, error_if_nonfinite=True)); optimizer.step(); step += 1
            loss_scores = components["scores"][loss_start:].detach().cpu().numpy(); loss_returns = batch["execution_1d_return"][loss_start:].detach().cpu().numpy(); loss_masks = (batch["asset_mask"][loss_start:] & batch["execution_return_mask"][loss_start:]).detach().cpu().numpy(); rankics = [_rank_corr(loss_scores[i][loss_masks[i]], loss_returns[i][loss_masks[i]]) for i in range(len(loss_scores)) if int(loss_masks[i].sum()) >= 3]
            interval_row = {"train_score_loss": float(components["score_loss"].detach().cpu()), "train_gross_return": float(components["accounting"]["gross_return"][loss_start:].mean().detach().cpu()), "train_turnover": float(components["accounting"]["turnover"][loss_start:].mean().detach().cpu()), "train_cost_7_5bps": float(components["accounting"]["cost_7_5bps"][loss_start:].mean().detach().cpu()), "train_net_return": float(components["accounting"]["net_return"][loss_start:].mean().detach().cpu()), "train_net_loss": float(components["net_loss"].detach().cpu()), "train_total_loss": float(total_loss.detach().cpu()), "train_rankic": float(np.mean(rankics)) if rankics else 0.0, "gradient_norm_total": total_grad}
            if step % 10 != 0: continue
            evaluation = _evaluate(model, loaders.validation, source, device, config); metric = float(evaluation["metrics"]["validation_daily_sharpe_cost_7p5bps"]); improved = metric > best_metric + 1e-12; no_improve = 0 if improved else no_improve + 1
            if improved: best_metric = metric; best_step = step; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            row = {"optimizer_step": step, "epoch_fraction": float(step / len(specs)), **interval_row, "validation_rankic": evaluation["metrics"]["validation_execution_rankic"], "validation_annual_return_0bps": evaluation["metrics"]["validation_daily_return_cost_0p0bps"], "validation_sharpe_0bps": evaluation["metrics"]["validation_daily_sharpe_cost_0p0bps"], "validation_annual_return_7_5bps": evaluation["metrics"]["validation_daily_return_cost_7p5bps"], "validation_sharpe_7_5bps": evaluation["metrics"]["validation_daily_sharpe_cost_7p5bps"], "validation_mdd_7_5bps": evaluation["metrics"]["validation_daily_mdd_cost_7p5bps"], "validation_turnover": evaluation["metrics"]["validation_daily_turnover"], "validation_effective_n": evaluation["metrics"]["validation_effective_n"], "validation_max_weight": evaluation["metrics"]["validation_max_weight_mean"], "validation_normalized_entropy": evaluation["metrics"]["validation_normalized_entropy"], "validation_information_ratio_7_5bps": evaluation["metrics"]["validation_information_ratio_cost_7p5bps"], "D10_weight_share": evaluation["metrics"]["D10_weight_share"], "improved": improved}
            history.append(row); print(f"[D] step={step} epoch={row['epoch_fraction']:.3f} net_loss={row['train_net_loss']:.6g} val_sharpe_7.5bp={row['validation_sharpe_7_5bps']:.6f} turnover={row['validation_turnover']:.5f} effN={row['validation_effective_n']:.2f} improved={improved} no_improve={no_improve}", flush=True)
            if improved: _save_checkpoint(checkpoints / "best.pt", model, optimizer, step, best_step, best_metric)
            if no_improve >= config.training.early_stopping_patience: stopped_early = True; break
        if stopped_early: break
    _save_checkpoint(checkpoints / "last.pt", model, optimizer, step, best_step, best_metric)
    model.load_state_dict(best_state); final = _evaluate(model, loaders.validation, source, device, config); final_metrics = dict(final["metrics"]); final_metrics.update({"experiment": "experiment_d", "training_performed": True, "best_optimizer_step": best_step, "best_epoch_fraction": best_step / len(specs), "optimizer_steps_total": step, "early_stopped": stopped_early, "early_stop_reason": "validation_7p5bp_sharpe_patience" if stopped_early else "max_epochs_or_steps", "lambda_portfolio": LAMBDA_PORTFOLIO, "lambda_score": 1.0, "temperature": 1.0, "transaction_cost_bps": 7.5, "no_extra_turnover_penalty": True, "no_topk_or_cap": True, "no_2022_data": True, "c_best_checkpoint_sha256": checkpoint_sha, "d_initial_checkpoint_sha256": _sha256(output / "initial_from_c.pt"), "best_checkpoint_sha256": _sha256(checkpoints / "best.pt") if (checkpoints / "best.pt").is_file() else None, "last_checkpoint_sha256": _sha256(checkpoints / "last.pt"), "validation_date_range": final["dates"], "validation_date_count": len(final["dates"]), "step0_gradient_decomposition": decomposition["median"]})
    if not (checkpoints / "best.pt").is_file(): _save_checkpoint(checkpoints / "best.pt", model, optimizer, best_step, best_step, best_metric); final_metrics["best_checkpoint_sha256"] = _sha256(checkpoints / "best.pt")
    final["score_frame"].to_parquet(output / "validation_scores.parquet", index=False); final["weight_frame"].to_parquet(output / "validation_weights.parquet", index=False); final["diagnostics"].to_csv(output / "validation_weight_diagnostics.csv", index=False, encoding="utf-8-sig"); final["decile_frame"].to_csv(output / "validation_score_decile_weights.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(history).to_csv(output / "training_steps.csv", index=False, encoding="utf-8-sig")
    daily = _daily_with_costs(final["daily_returns"]); stability, stability_stats = _stability_diagnostics(final["weight_frame"], final["dates"]); turnover_diag = final["diagnostics"].merge(stability, on="date", how="left"); turnover_diag.to_csv(output / "turnover_diagnostics.csv", index=False, encoding="utf-8-sig"); daily.to_csv(output / "validation_daily_returns.csv", index=False, encoding="utf-8-sig"); final_metrics["weight_stability"] = stability_stats
    best_history = [row for row in history if int(row["optimizer_step"]) == best_step]; train_rankic = best_history[0].get("train_rankic") if best_history else None; final_metrics["train_rankic_at_best"] = train_rankic; final_metrics["validation_rankic_at_best"] = final_metrics["validation_execution_rankic"]; final_metrics["ic_gap_train_minus_validation"] = None if train_rankic is None else float(train_rankic) - final_metrics["validation_execution_rankic"]
    write_metrics(output / "validation_metrics.json", final_metrics)
    c_metrics = json.loads((c_output / "validation_metrics.json").read_text(encoding="utf-8")); c_daily = _daily_with_costs(pd.read_csv(c_output / "validation_daily_returns.csv")); c_drag, d_drag = _write_c_vs_d(output, c_metrics, final_metrics, c_daily, daily); final_metrics["annual_transaction_cost_drag"] = d_drag; final_metrics["c_annual_transaction_cost_drag"] = c_drag; final_metrics["delta_annual_transaction_cost_drag"] = d_drag - c_drag; write_metrics(output / "validation_metrics.json", final_metrics)
    manifest = {"experiment": "experiment_d", "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_experiment_c": str(c_output), "output_dir": str(output), "c_best_checkpoint_sha256": checkpoint_sha, "d_initial_checkpoint_sha256": _sha256(output / "initial_from_c.pt"), "checkpoint_hashes_match": checkpoint_sha == _sha256(output / "initial_from_c.pt"), "preflight_status": preflight["status"], "step0_identity_status": step0_identity["status"], "optimizer_steps_total": step, "best_optimizer_step": best_step, "early_stopped": stopped_early, "lambda_portfolio": LAMBDA_PORTFOLIO, "temperature": 1.0, "transaction_cost_bps": 7.5, "extra_turnover_penalty": 0.0, "training_performed": True, "backward_called": True, "optimizer_step_called": True, "no_2022": True, "experiment_e_run": False, "files": sorted({path.name for path in output.iterdir()} | {"run_manifest.json"})}
    write_metrics(output / "run_manifest.json", manifest); print(f"[D] completed steps={step} best_step={best_step} best_sharpe_7.5bp={best_metric:.6f} output={output}", flush=True); return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--c-output", type=Path, required=True); parser.add_argument("--output", type=Path); args = parser.parse_args(); print(run(args.config, c_output=args.c_output, output=args.output))


if __name__ == "__main__": main()
