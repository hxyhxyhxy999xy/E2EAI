"""Diagnose frozen train-only factors on train and purged validation dates only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from e2eai.config import load_config
from e2eai.data.loaders import _split_indices_with_metadata
from e2eai.data.panel import AlphaPanelDataset
from e2eai.workflows import load_configured_data


def _rank_ic(values: np.ndarray, returns: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(returns)
    if int(valid.sum()) < 3:
        return float("nan")
    x = rankdata(values[valid], method="average")
    y = rankdata(returns[valid], method="average")
    denominator = float(np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2)))
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denominator) if denominator > 1e-12 else float("nan")


def _statistics(values: np.ndarray, prefix: str) -> dict[str, float | int | None]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {
            f"{prefix}_mean_rankic": None,
            f"{prefix}_rankic_std": None,
            f"{prefix}_icir": None,
            f"{prefix}_positive_ic_ratio": None,
            f"{prefix}_valid_date_count": 0,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    return {
        f"{prefix}_mean_rankic": mean,
        f"{prefix}_rankic_std": std,
        f"{prefix}_icir": float(mean / std) if std > 1e-12 else 0.0,
        f"{prefix}_positive_ic_ratio": float((clean > 0.0).mean()),
        f"{prefix}_valid_date_count": int(clean.size),
    }


def _summary(series: pd.Series) -> dict[str, float | int | None]:
    clean = series.dropna().astype(float)
    if clean.empty:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/daily_strategy.yaml"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/daily_strategy_diagnostics/factor_diagnostics_trainonly64.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("output/daily_strategy_diagnostics/factor_diagnostics_trainonly64_summary.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.data.daily_validation_only:
        raise ValueError("Diagnostics require data.daily_validation_only=true")
    source, synthetic = load_configured_data(config)
    if synthetic:
        raise RuntimeError("Train-only factor diagnostics require the configured real panel")
    dataset = AlphaPanelDataset(
        source, config.model.horizons, config.data, execution_horizon=config.model.execution_horizon
    )
    train_indices, validation_indices, test_indices, split_metadata = _split_indices_with_metadata(config, dataset)
    if test_indices:
        raise RuntimeError("2022 test split must be empty for this diagnostic")

    panel = dataset.panel
    if panel.execution_returns is None or panel.factor_valid_mask is None:
        raise ValueError("Execution returns and factor validity mask are required")
    factor_count = len(panel.alpha_names)
    if factor_count != 64:
        raise ValueError(f"Expected 64 frozen factors, found {factor_count}")
    all_indices = train_indices + validation_indices
    daily_ic = np.full((len(all_indices), factor_count), np.nan, dtype=np.float64)
    cross_sectional_mean = np.full_like(daily_ic, np.nan)
    cross_sectional_std = np.full_like(daily_ic, np.nan)
    years = np.asarray([dataset.dates[index].year for index in all_indices], dtype=np.int16)

    for row, date_index in enumerate(all_indices):
        positions = dataset._selected_assets(date_index)
        execution = np.asarray(panel.execution_returns[date_index, positions], dtype=np.float64)
        factors = np.asarray(panel.alpha_tensor[date_index, :, positions], dtype=np.float64)
        validity = np.asarray(panel.factor_valid_mask[date_index, :, positions], dtype=bool)
        if factors.shape != (factor_count, len(positions)):
            factors = factors.T
            validity = validity.T
        for factor in range(factor_count):
            values = factors[factor]
            usable = validity[factor] & np.isfinite(values)
            if usable.any():
                cross_sectional_mean[row, factor] = float(values[usable].mean())
                cross_sectional_std[row, factor] = float(values[usable].std(ddof=0))
            daily_ic[row, factor] = _rank_ic(
                np.where(validity[factor], values, np.nan), execution
            )

    train_count = len(train_indices)
    train_ic = daily_ic[:train_count]
    validation_ic = daily_ic[train_count:]
    rows: list[dict[str, Any]] = []
    for factor, name in enumerate(panel.alpha_names):
        record: dict[str, Any] = {"factor_name": name, "factor_index": factor}
        record.update(_statistics(train_ic[:, factor], "train"))
        record.update(_statistics(validation_ic[:, factor], "validation"))
        for year in (2018, 2019, 2020, 2021):
            record[f"{year}_mean_rankic"] = _statistics(
                daily_ic[years == year, factor], str(year)
            )[f"{year}_mean_rankic"]
        record["train_std_t_cross_sectional_mean"] = float(
            np.nanstd(cross_sectional_mean[:train_count, factor], ddof=0)
        )
        record["train_mean_t_cross_sectional_std"] = float(
            np.nanmean(cross_sectional_std[:train_count, factor])
        )
        record["validation_std_t_cross_sectional_mean"] = float(
            np.nanstd(cross_sectional_mean[train_count:, factor], ddof=0)
        )
        record["validation_mean_t_cross_sectional_std"] = float(
            np.nanmean(cross_sectional_std[train_count:, factor])
        )
        rows.append(record)
    diagnostics = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    pooling_std = diagnostics["train_std_t_cross_sectional_mean"]
    summary = {
        "panel": str(config.data.path),
        "selection_manifest": config.data.factor_selection_manifest_path,
        "selection_target": "execution_1d_return = VWAP[t+2] / VWAP[t+1] - 1",
        "split_metadata": split_metadata,
        "2022_test_split_not_evaluated": True,
        "factor_count": factor_count,
        "train_mean_rankic": _summary(diagnostics["train_mean_rankic"]),
        "validation_mean_rankic": _summary(diagnostics["validation_mean_rankic"]),
        "annual_mean_rankic": {
            str(year): _summary(diagnostics[f"{year}_mean_rankic"])
            for year in (2018, 2019, 2020, 2021)
        },
        "mean_pooling_diagnostic": {
            "median_std_t_cross_sectional_mean": float(pooling_std.median()),
            "max_std_t_cross_sectional_mean": float(pooling_std.max()),
            "median_mean_t_cross_sectional_std": float(
                diagnostics["train_mean_t_cross_sectional_std"].median()
            ),
            "suspected_degenerate": bool((pooling_std < 1e-6).mean() >= 0.95),
            "interpretation": (
                "True means the daily cross-sectional factor mean is effectively constant over time; "
                "mean-only factor pooling would then be uninformative."
            ),
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
