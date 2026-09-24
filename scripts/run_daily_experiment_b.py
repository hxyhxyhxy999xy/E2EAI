"""Run Experiment B: continuous z-score + masked-softmax mapping.

Experiment B is deliberately evaluation-only.  It reloads the exact
Experiment A checkpoint, regenerates the raw validation scores, verifies them
against A's saved score file, and applies a portfolio mapping outside the
trained model.  No optimizer, back-propagation, top-k, cap, clipping, or 2022
data is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Make ``python scripts/run_daily_experiment_b.py`` work from a source checkout
# without requiring callers to set PYTHONPATH explicitly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.evaluation.continuous_portfolio import continuous_long_only_weights
from e2eai.evaluation.daily_validation import _simulate_daily_weight_path, daily_path_metrics
from e2eai.training.trainer import E2EAITrainer
from e2eai.utils.reproducibility import seed_everything
from e2eai.workflows import load_configured_data, write_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token(cost: float) -> str:
    return str(float(cost)).replace(".", "p")


def _canonical_dates(values: pd.Series | list[Any]) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.strftime("%Y-%m-%d")


def _canonical_assets(values: pd.Series) -> pd.Series:
    """Normalize CSV-parsed numeric IDs to the panel's six-digit strings."""
    def one(value: Any) -> str:
        text = str(value)
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(6) if text.isdigit() else text
    return values.map(one).astype(str)


def _identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset_id", "raw_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"score file is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = _canonical_dates(result["date"])
    result["asset_id"] = _canonical_assets(result["asset_id"])
    result["raw_score"] = pd.to_numeric(result["raw_score"], errors="coerce")
    return result.sort_values(["date", "asset_id"], kind="stable").reset_index(drop=True)


def _score_identity(a_scores: pd.DataFrame, regenerated: pd.DataFrame) -> dict[str, Any]:
    left_raw = a_scores.copy()
    right_raw = regenerated.copy()
    for raw in (left_raw, right_raw):
        raw["date"] = _canonical_dates(raw["date"])
        raw["asset_id"] = _canonical_assets(raw["asset_id"])
    left = _identity_frame(a_scores)
    right = _identity_frame(regenerated)
    axis_columns = ["date", "asset_id"]
    same_shape = left.shape == right.shape
    # ``DataFrame.equals`` is dtype-sensitive (the saved CSV may parse numeric
    # asset IDs as integers while the regenerated loader returns strings).
    # Compare canonical string values so the identity check tests the actual
    # date/asset axis rather than CSV parser dtypes.
    left_axis = left[axis_columns].astype(str).to_numpy()
    right_axis = right[axis_columns].astype(str).to_numpy()
    same_axis = same_shape and bool(np.array_equal(left_axis, right_axis))
    same_axis_original = same_shape and bool(
        np.array_equal(
            left_raw[axis_columns].to_numpy(),
            right_raw[axis_columns].to_numpy(),
        )
    )
    finite_left = np.isfinite(left["raw_score"].to_numpy(dtype=np.float64))
    finite_right = np.isfinite(right["raw_score"].to_numpy(dtype=np.float64))
    finite_match = bool(same_shape and np.array_equal(finite_left, finite_right))
    max_abs_diff = None
    if same_shape:
        values_left = left["raw_score"].to_numpy(dtype=np.float64)
        values_right = right["raw_score"].to_numpy(dtype=np.float64)
        both = np.isfinite(values_left) & np.isfinite(values_right)
        if bool(both.any()):
            max_abs_diff = float(np.max(np.abs(values_left[both] - values_right[both])))
        elif bool(np.array_equal(finite_left, finite_right)):
            max_abs_diff = 0.0
    passed = bool(same_shape and same_axis and finite_match and (max_abs_diff is not None) and max_abs_diff <= 1e-6)
    return {
        "status": "PASS" if passed else "FAIL",
        "date_axis": "sorted canonical YYYY-MM-DD",
        "asset_axis": "sorted asset_id",
        "columns_compared": axis_columns + ["raw_score"],
        "shape_a": list(left.shape),
        "shape_b": list(right.shape),
        "same_shape": same_shape,
        "same_date_asset_axis": same_axis,
        "same_date_asset_axis_original_order": same_axis_original,
        "finite_mask_match": finite_match,
        "max_abs_diff": max_abs_diff,
        "tolerance": 1e-6,
        "a_date_range": [str(left["date"].min()), str(left["date"].max())] if len(left) else None,
        "b_date_range": [str(right["date"].min()), str(right["date"].max())] if len(right) else None,
        "a_rows": int(len(left)),
        "b_rows": int(len(right)),
    }


def _build_weights(score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float]]]:
    score_frame = score_frame.copy()
    score_frame["date"] = _canonical_dates(score_frame["date"])
    score_frame["asset_id"] = score_frame["asset_id"].astype(str)
    score_frame["raw_score"] = pd.to_numeric(score_frame["raw_score"], errors="coerce")
    score_frame["execution_1d_return"] = pd.to_numeric(
        score_frame.get("execution_1d_return", pd.Series(index=score_frame.index, dtype=float)),
        errors="coerce",
    )
    weight_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    target_weights: list[dict[str, float]] = []

    for date, group in score_frame.groupby("date", sort=True):
        raw = group["raw_score"].to_numpy(dtype=np.float64)
        valid = np.isfinite(raw)
        weights, normalized, mean, std = continuous_long_only_weights(
            raw, valid, temperature=1.0, eps=1e-6
        )
        assets = group["asset_id"].tolist()
        target = {
            str(asset): float(weight)
            for asset, weight, ok in zip(assets, weights, valid)
            if bool(ok)
        }
        target_weights.append(target)
        positive = weights[valid]
        hhi = float(np.square(positive).sum())
        effective_n = float(1.0 / max(hhi, 1e-12))
        entropy = float(-np.sum(positive * np.log(np.maximum(positive, 1e-300))))
        normalized_entropy = float(entropy / np.log(len(positive))) if len(positive) > 1 else 0.0
        quantiles = np.quantile(positive, [0.50, 0.75, 0.90, 0.95, 0.99])
        diagnostic_rows.append(
            {
                "date": date,
                "valid_asset_count": int(valid.sum()),
                "score_mean": mean,
                "score_std": std,
                "effective_n": effective_n,
                "hhi": hhi,
                "max_weight": float(positive.max()),
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "weight_p50": float(quantiles[0]),
                "weight_p75": float(quantiles[1]),
                "weight_p90": float(quantiles[2]),
                "weight_p95": float(quantiles[3]),
                "weight_p99": float(quantiles[4]),
                "count_weight_gt_0p001": int((positive > 0.001).sum()),
                "count_weight_gt_0p0025": int((positive > 0.0025).sum()),
                "count_weight_gt_0p005": int((positive > 0.005).sum()),
                "count_weight_gt_0p01": int((positive > 0.01).sum()),
                "count_weight_gt_0p02": int((positive > 0.02).sum()),
                "n_weight_gt_0_1pct": int((positive > 0.001).sum()),
                "n_weight_gt_0_25pct": int((positive > 0.0025).sum()),
                "n_weight_gt_0_5pct": int((positive > 0.005).sum()),
                "n_weight_gt_1pct": int((positive > 0.01).sum()),
                "n_weight_gt_2pct": int((positive > 0.02).sum()),
            }
        )
        for asset, raw_value, z_value, weight, ok, execution in zip(
            assets,
            raw,
            normalized,
            weights,
            valid,
            group["execution_1d_return"].to_numpy(dtype=np.float64),
        ):
            weight_rows.append(
                {
                    "date": date,
                    "asset": str(asset),
                    "raw_score": float(raw_value) if np.isfinite(raw_value) else np.nan,
                    "normalized_score": float(z_value) if bool(ok) else np.nan,
                    "weight": float(weight),
                    "valid_asset": bool(ok),
                    "execution_1d_return": float(execution) if np.isfinite(execution) else np.nan,
                }
            )

    return pd.DataFrame(weight_rows), pd.DataFrame(diagnostic_rows), target_weights


def _benchmark_for_dates(source: Any, dates: list[str]) -> np.ndarray:
    benchmark = getattr(source, "benchmark_returns", None)
    panel_dates = getattr(source, "dates", None)
    if benchmark is None or panel_dates is None:
        raise ValueError("Experiment B requires the CSI500 benchmark path")
    values = np.asarray(benchmark)
    if values.ndim != 1:
        raise ValueError("Experiment B requires a one-dimensional execution benchmark")
    lookup = {pd.Timestamp(date).strftime("%Y-%m-%d"): float(value) for date, value in zip(panel_dates, values)}
    missing = [date for date in dates if date not in lookup]
    if missing:
        raise ValueError(f"Benchmark is missing validation dates, first missing={missing[:3]}")
    return np.asarray([lookup[date] for date in dates], dtype=np.float64)


def _path_dates(score_frame: pd.DataFrame) -> list[str]:
    dates: list[str] = []
    for date, group in score_frame.groupby("date", sort=True):
        if np.isfinite(group["raw_score"].to_numpy(dtype=np.float64)).any() and np.isfinite(
            pd.to_numeric(group["execution_1d_return"], errors="coerce").to_numpy(dtype=np.float64)
        ).any():
            dates.append(str(date))
    return dates


def _a_topk_normalized_entropy(score_frame: pd.DataFrame, ratio: float = 0.20) -> float:
    """Reconstruct A's Top20 EW normalized entropy for the A/B structure row."""
    values: list[float] = []
    frame = _identity_frame(score_frame)
    for _, group in frame.groupby("date", sort=True):
        n = int(np.isfinite(group["raw_score"].to_numpy(dtype=np.float64)).sum())
        if n <= 1:
            values.append(0.0)
            continue
        k = max(1, min(n, int(np.ceil(n * ratio))))
        values.append(float(np.log(k) / np.log(n)))
    return float(np.mean(values)) if values else 0.0


def _performance(
    target_weights: list[dict[str, float]],
    execution_returns: list[dict[str, float]],
    benchmark: np.ndarray,
    costs: list[float],
    *,
    annual_trading_days: int,
    holding_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    paths: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, Any] = {}
    for cost in costs:
        token = _token(cost)
        path = _simulate_daily_weight_path(
            target_weights,
            execution_returns,
            cost_bps=float(cost),
            holding_threshold=holding_threshold,
        )
        paths[token] = path
        summary = daily_path_metrics(
            path,
            annual_trading_days=annual_trading_days,
            benchmark_returns=benchmark,
        )
        for key, value in summary.items():
            metrics[f"validation_{key.removeprefix('validation_')}_cost_{token}bps"] = value
    primary = paths[_token(7.5)]
    primary_summary = daily_path_metrics(
        primary,
        annual_trading_days=annual_trading_days,
        benchmark_returns=benchmark,
    )
    metrics.update(primary_summary)
    metrics["validation_daily_return_7_5bps"] = primary_summary["validation_daily_return"]
    metrics["validation_daily_sharpe_7_5bps"] = primary_summary["validation_daily_sharpe"]

    frame = pd.DataFrame(
        {
            "gross_return": primary["gross_returns"],
            "one_way_turnover": primary["turnovers"],
            "benchmark_return": benchmark,
        }
    )
    for cost in costs:
        token = _token(cost)
        frame[f"net_return_{token}bps"] = paths[token]["net_returns"]
    frame["active_return_7p5bps"] = frame["net_return_7p5bps"] - frame["benchmark_return"]
    return metrics, frame


def _write_a_vs_b(
    destination: Path,
    a_metrics: dict[str, Any],
    b_metrics: dict[str, Any],
    factor_metrics: dict[str, Any],
    costs: list[float],
) -> None:
    """Write a three-way comparison: Experiment A, B, and Factor Mean EW."""
    rows: list[dict[str, Any]] = []

    def append_row(cost: float | str, metric: str, a_value: Any, b_value: Any, factor_value: Any) -> None:
        rows.append(
            {
                "cost_bps": cost,
                "metric": metric,
                "A": a_value,
                "B": b_value,
                "FactorMean": factor_value,
                "B_minus_A": None if a_value is None or b_value is None else float(b_value) - float(a_value),
                "B_minus_FactorMean": None
                if factor_value is None or b_value is None
                else float(b_value) - float(factor_value),
            }
        )

    for cost in costs:
        token = _token(cost)
        pairs = {
            "annual_return": f"validation_daily_return_cost_{token}bps",
            "sharpe": f"validation_daily_sharpe_cost_{token}bps",
            "annual_excess_return": f"validation_active_annual_excess_return_cost_{token}bps",
            "benchmark_annual_return": f"validation_benchmark_annual_return_cost_{token}bps",
            "information_ratio": f"validation_information_ratio_cost_{token}bps",
            "tracking_error": f"validation_tracking_error_cost_{token}bps",
            "mdd": f"validation_daily_mdd_cost_{token}bps",
            "active_max_drawdown": f"validation_active_max_drawdown_cost_{token}bps",
            "active_win_rate": f"validation_active_win_rate_cost_{token}bps",
            "active_cumulative_return": f"validation_active_cumulative_return_cost_{token}bps",
        }
        for metric, key in pairs.items():
            append_row(cost, metric, a_metrics.get(key), b_metrics.get(key), factor_metrics.get(key))
    structure = {
        "daily_turnover": "validation_daily_turnover",
        "effective_n": "validation_effective_n",
        "max_weight": "validation_max_weight",
        "hhi": "validation_hhi",
        "normalized_entropy": "validation_normalized_entropy",
        "execution_rankic": "validation_execution_rankic",
    }
    for metric, key in structure.items():
        append_row("all", metric, a_metrics.get(key), b_metrics.get(key), factor_metrics.get(key))
    pd.DataFrame(rows).to_csv(destination / "A_vs_B.csv", index=False, encoding="utf-8-sig")
    b_rankic = b_metrics.get("validation_execution_rankic")
    a_rankic = a_metrics.get("validation_execution_rankic")
    lines = [
        "# Experiment A vs Experiment B",
        "",
        "Experiment B reuses the exact Experiment A checkpoint and raw validation scores. It changes only the portfolio mapping to cross-sectional z-score followed by temperature-1 masked softmax. The same table also includes the existing Factor Mean equal-weight baseline for every cost scenario.",
        "",
        f"- Validation RankIC A: `{a_rankic}`; B: `{b_rankic}`; absolute difference: `{abs(float(a_rankic) - float(b_rankic)) if a_rankic is not None and b_rankic is not None else 'UNAVAILABLE'}`.",
        f"- A best epoch: `{a_metrics.get('best_epoch')}`; A completed epochs: `{a_metrics.get('epochs_completed')}`.",
        "- B optimizer steps: `0`; B training: `false`; C/D and later experiments: not run.",
        "- Validation period is 2021 only; no 2022 dates were loaded or evaluated.",
        "",
        "| Cost (bp) | Metric | A | B | Factor Mean EW | B − A | B − FactorMean |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in pd.read_csv(destination / "A_vs_B.csv").to_dict("records"):
        if row["cost_bps"] == "all":
            continue
        lines.append(
            f"| {row['cost_bps']} | {row['metric']} | {row['A']} | {row['B']} | "
            f"{row['FactorMean']} | {row['B_minus_A']} | {row['B_minus_FactorMean']} |"
        )
    lines.extend(
        [
            "",
            "## Structure",
            "",
            "The B mapping is expected to be diffuse because every finite validation asset receives a positive weight. The CSV contains daily effective N, HHI, entropy, quantiles, and threshold counts for audit.",
        ]
    )
    (destination / "A_vs_B.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    *,
    output: Path | None = None,
    a_output: Path | None = None,
    checkpoint: Path | None = None,
) -> Path:
    # Explicit import keeps this script usable even if a static checker sees
    # the explanatory import comment above.
    from e2eai.models.e2eai import E2EAIModel

    a_config = load_config(config_path)
    if a_config.experiment_line != "daily_strategy" or a_config.model.strategy_head != "simple_mlp":
        raise ValueError("Experiment B requires the daily-core Experiment A config")
    if a_config.model.horizons != [2] or a_config.model.execution_horizon != 2:
        raise ValueError("Experiment B requires h=2 execution-only validation")
    if a_config.data.test_start is not None or a_config.data.test_end is not None:
        raise ValueError("Experiment B must not configure an OOS/test period")
    a_destination = a_output or Path(a_config.logging.output_dir)
    b_destination = output or (a_destination.parent / "experiment_b")
    b_destination.mkdir(parents=True, exist_ok=True)
    for child in b_destination.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    checkpoint_path = checkpoint or (a_destination / "best.pt")
    a_scores_path = a_destination / "validation_scores.csv"
    a_metrics_path = a_destination / "validation_metrics.json"
    if not checkpoint_path.is_file() or not a_scores_path.is_file() or not a_metrics_path.is_file():
        raise FileNotFoundError("Experiment A checkpoint, scores, and metrics are required")
    checkpoint_sha = _sha256(checkpoint_path)
    a_metrics = json.loads(a_metrics_path.read_text(encoding="utf-8"))
    factor_mean_metrics_path = a_destination.parent / "factor_mean_baseline" / "validation_metrics.json"
    if not factor_mean_metrics_path.is_file():
        raise FileNotFoundError(
            "Factor Mean equal-weight baseline metrics are required for every experiment comparison: "
            f"{factor_mean_metrics_path}"
        )
    factor_mean_metrics = json.loads(factor_mean_metrics_path.read_text(encoding="utf-8"))

    # Loading and scoring are inference-only.  create_optimizer=False is an
    # explicit guard against accidental B training.
    seed_everything(a_config.seed)
    source, synthetic = load_configured_data(a_config)
    loaders = build_dataloaders(a_config, source)
    model = E2EAIModel(a_config.model, a_config.ablation)
    trainer = E2EAITrainer(model, a_config, create_optimizer=False)
    try:
        trainer.load_checkpoint(checkpoint_path, load_optimizer=False, restore_rng=False)
        regenerated_records, regenerated_checksum = trainer.collect_raw_scores(loaders.validation)
    finally:
        trainer.close()
    regenerated = pd.DataFrame(regenerated_records)
    formal_a = pd.read_csv(a_scores_path)
    identity = _score_identity(formal_a, regenerated)
    identity.update(
        {
            "checkpoint_sha256_a": checkpoint_sha,
            "checkpoint_sha256_b": checkpoint_sha,
            "raw_score_sha256_a_file": a_metrics.get("raw_score_sha256"),
            "raw_score_sha256_regenerated": regenerated_checksum,
            "raw_score_sha256_match": bool(a_metrics.get("raw_score_sha256") in {None, regenerated_checksum}),
            "no_optimizer_steps": True,
            "model_loaded_from": str(checkpoint_path),
        }
    )
    if identity["status"] != "PASS":
        write_metrics(b_destination / "score_identity_check.json", identity)
        raise RuntimeError(f"A/B score identity check failed: {identity}")
    write_metrics(b_destination / "score_identity_check.json", identity)

    score_frame = regenerated.copy()
    score_frame["date"] = _canonical_dates(score_frame["date"])
    weight_frame, diagnostics, target_weights = _build_weights(score_frame)
    dates = sorted(diagnostics["date"].tolist())
    if any(str(date).startswith("2022") for date in dates):
        raise RuntimeError("Experiment B unexpectedly contains 2022 validation dates")
    path_dates = _path_dates(score_frame)
    if dates != path_dates:
        raise RuntimeError("Some validation dates have no finite execution return; refusing an implicit date drop")
    execution_returns = []
    for date, group in score_frame.groupby("date", sort=True):
        execution_returns.append(
            {
                str(asset): float(value)
                for asset, value in zip(group["asset_id"], pd.to_numeric(group["execution_1d_return"], errors="coerce"))
                if np.isfinite(value)
            }
        )
    benchmark = _benchmark_for_dates(source, dates)
    costs = [float(value) for value in a_config.evaluation.cost_scenarios_bps]
    if costs != [0.0, 5.0, 7.5, 10.0]:
        raise ValueError("Experiment B requires cost scenarios 0/5/7.5/10 bp")
    performance, daily_returns = _performance(
        target_weights,
        execution_returns,
        benchmark,
        costs,
        annual_trading_days=a_config.evaluation.annual_trading_days,
        holding_threshold=a_config.evaluation.holding_threshold,
    )
    daily_returns.insert(0, "date", dates)
    daily_returns.to_csv(b_destination / "validation_daily_returns.csv", index=False, encoding="utf-8-sig")
    weight_frame.to_parquet(b_destination / "validation_weights.parquet", index=False)
    diagnostics.to_csv(b_destination / "validation_weight_diagnostics.csv", index=False, encoding="utf-8-sig")

    diag_values = diagnostics
    b_metrics: dict[str, Any] = {
        "experiment": "experiment_b",
        "mapping": "cross_sectional_zscore_then_masked_softmax",
        "temperature": 1.0,
        "zscore_eps": 1e-6,
        "allocation_constraints": {
            "topk": False,
            "single_stock_cap": False,
            "clipping": False,
            "threshold": False,
            "long_only": True,
        },
        "training_performed": False,
        "optimizer_steps": 0,
        "checkpoint_sha256": checkpoint_sha,
        "raw_score_sha256": regenerated_checksum,
        "identity_check": "PASS",
        "synthetic": synthetic,
        "oos_evaluated": False,
        "no_2022_data": True,
        "horizons_used": [2],
        "validation_date_range": [dates[0], dates[-1]],
        "validation_date_count": len(dates),
        "validation_execution_rankic": float(a_metrics.get("validation_execution_rankic", np.nan)),
        "validation_execution_icir": float(a_metrics.get("validation_execution_icir", np.nan)),
        "validation_execution_positive_ic_ratio": float(a_metrics.get("validation_execution_positive_ic_ratio", np.nan)),
        "validation_daily_turnover": float(performance["validation_daily_turnover"]),
        "validation_effective_n": float(diag_values["effective_n"].mean()),
        "validation_hhi": float(diag_values["hhi"].mean()),
        "validation_max_weight": float(diag_values["max_weight"].max()),
        "validation_entropy": float(diag_values["entropy"].mean()),
        "validation_normalized_entropy": float(diag_values["normalized_entropy"].mean()),
        "mean_effective_n_over_valid_count": float(
            (diag_values["effective_n"] / diag_values["valid_asset_count"].clip(lower=1)).mean()
        ),
        "validation_effective_n_median": float(diag_values["effective_n"].median()),
        "validation_effective_n_p10": float(diag_values["effective_n"].quantile(0.10)),
        "validation_effective_n_min": float(diag_values["effective_n"].min()),
        "validation_max_weight_mean": float(diag_values["max_weight"].mean()),
        "validation_max_weight_median": float(diag_values["max_weight"].median()),
        "validation_max_weight_p95": float(diag_values["max_weight"].quantile(0.95)),
        "validation_max_weight_max": float(diag_values["max_weight"].max()),
        "validation_mean_hhi": float(diag_values["hhi"].mean()),
        "validation_mean_normalized_entropy": float(diag_values["normalized_entropy"].mean()),
        "weight_quantiles_period_mean": {
            key: float(diag_values[key].mean())
            for key in ["weight_p50", "weight_p75", "weight_p90", "weight_p95", "weight_p99"]
        },
        "threshold_count_period_mean": {
            key: float(diag_values[key].mean())
            for key in [
                "count_weight_gt_0p001",
                "count_weight_gt_0p0025",
                "count_weight_gt_0p005",
                "count_weight_gt_0p01",
                "count_weight_gt_0p02",
            ]
        },
        "mean_n_weight_gt_0_1pct": float(diag_values["n_weight_gt_0_1pct"].mean()),
        "mean_n_weight_gt_0_25pct": float(diag_values["n_weight_gt_0_25pct"].mean()),
        "mean_n_weight_gt_0_5pct": float(diag_values["n_weight_gt_0_5pct"].mean()),
        "mean_n_weight_gt_1pct": float(diag_values["n_weight_gt_1pct"].mean()),
        "mean_n_weight_gt_2pct": float(diag_values["n_weight_gt_2pct"].mean()),
        "top10_max_weight_dates": diagnostics.nlargest(10, "max_weight")[
            ["date", "max_weight", "effective_n", "valid_asset_count"]
        ].to_dict("records"),
        "a_best_epoch": a_metrics.get("best_epoch"),
        "a_epochs_completed": a_metrics.get("epochs_completed"),
        "factor_mean_baseline_metrics": str(factor_mean_metrics_path),
        **performance,
    }
    write_metrics(b_destination / "validation_metrics.json", b_metrics)
    diagnostics.to_csv(b_destination / "validation_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    a_metrics_for_compare = dict(a_metrics)
    a_metrics_for_compare["validation_normalized_entropy"] = _a_topk_normalized_entropy(score_frame)
    factor_mean_for_compare = dict(factor_mean_metrics)
    # Factor Mean uses the same Top20 EW allocation in this daily core, so its
    # normalized entropy is reconstructed from the same valid-stock panel.
    factor_mean_for_compare["validation_normalized_entropy"] = _a_topk_normalized_entropy(score_frame)
    _write_a_vs_b(b_destination, a_metrics_for_compare, b_metrics, factor_mean_for_compare, costs)

    # A's pre-update model state was not persisted.  We intentionally do not
    # present a seed-only reconstruction as byte-identical initialization.
    write_metrics(
        b_destination / "untrained_a_diagnostic.json",
        {
            "status": "UNAVAILABLE",
            "reason": "Experiment A did not save its pre-update model state; seed=42 reconstruction cannot be proven byte-identical after the fact.",
            "reconstruction_attempted": False,
            "optimizer_steps": 0,
            "top20_definition": "A's saved topk_equal_weight ratio=0.20",
            "untrained_validation_rankic": "UNAVAILABLE",
            "untrained_7_5bps_sharpe": "UNAVAILABLE",
            "untrained_7_5bps_annual_return": "UNAVAILABLE",
            "untrained_turnover": "UNAVAILABLE",
            "epoch0_vs_untrained": "UNAVAILABLE",
        },
    )

    resolved = a_config.to_dict()
    resolved["experiment_b"] = {
        "source_experiment": "experiment_a",
        "checkpoint": str(checkpoint_path),
        "mapping": "cross_sectional_zscore_then_masked_softmax",
        "temperature": 1.0,
        "zscore_eps": 1e-6,
        "use_raw_score_softmax": False,
        "topk": False,
        "single_stock_cap": False,
        "clipping": False,
        "training": False,
        "optimizer_steps": 0,
        "validation_only": True,
        "validation_period": [dates[0], dates[-1]],
    }
    (b_destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "experiment": "experiment_b",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(config_path),
        "source_experiment_output": str(a_destination),
        "output_dir": str(b_destination),
        "checkpoint_sha256": checkpoint_sha,
        "identity_check": str(b_destination / "score_identity_check.json"),
        "optimizer_steps": 0,
        "training_performed": False,
        "no_2022_data": True,
        "no_c_or_later": True,
        "files": sorted({path.name for path in b_destination.iterdir()} | {"run_manifest.json"}),
    }
    write_metrics(b_destination / "run_manifest.json", manifest)
    return b_destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment A daily-core config")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--a-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    print(run(args.config, output=args.output, a_output=args.a_output, checkpoint=args.checkpoint))


if __name__ == "__main__":
    main()
