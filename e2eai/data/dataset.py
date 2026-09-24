"""Cross-sectional CSV/Parquet dataset and synthetic signal generator."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from e2eai.data.preprocessing import impute_factors
from e2eai.data.schema import (
    benchmark_columns,
    infer_factor_columns,
    return_columns,
    validate_long_frame,
)


def load_market_frame(path: str | Path) -> pd.DataFrame:
    """Load a long-form market dataset from CSV or Parquet."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError("Data path must end in .csv, .parquet, or .pq")


class E2EAIDataset(Dataset[dict[str, Any]]):
    """One item per decision-date cross-section.

    Returned factor shape is ``[N_t,M]`` and return shape is ``[K,N_t]``.
    Missing labels are represented by zero plus a separate boolean mask.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        horizons: Sequence[int],
        factor_columns: Sequence[str] | None = None,
        *,
        date_column: str = "date",
        asset_column: str = "asset_id",
        industry_column: str = "industry_id",
        factor_prefix: str = "factor_",
        return_prefix: str = "forward_return_",
        benchmark_prefix: str = "benchmark_return_",
        execution_horizon: int | None = None,
        execution_return_column: str = "execution_1d_return",
        missing_factor_strategy: str = "cross_sectional_median",
    ) -> None:
        self.horizons = tuple(int(value) for value in horizons)
        self.execution_horizon = execution_horizon
        self.execution_return_column = execution_return_column
        self.factor_columns = infer_factor_columns(frame, factor_columns, factor_prefix)
        label_horizons = [
            horizon for horizon in self.horizons if horizon != self.execution_horizon
        ]
        self.label_horizons = tuple(label_horizons)
        self.return_columns = return_columns(self.label_horizons, return_prefix)
        self.benchmark_columns = benchmark_columns(self.horizons, benchmark_prefix)
        self.date_column = date_column
        self.asset_column = asset_column
        self.industry_column = industry_column
        working = frame.copy()
        working[date_column] = pd.to_datetime(working[date_column], errors="raise")
        validate_long_frame(
            working,
            date_column,
            asset_column,
            industry_column,
            self.factor_columns,
            self.return_columns,
        )
        if execution_horizon is not None and execution_return_column not in working.columns:
            raise ValueError(
                f"Dataset is missing execution-return column {execution_return_column!r}"
            )
        groups: list[pd.DataFrame] = []
        for _, cross_section in working.sort_values([date_column, asset_column]).groupby(
            date_column, sort=True
        ):
            cleaned = impute_factors(
                cross_section,
                self.factor_columns,
                missing_factor_strategy,
            )
            if len(cleaned) > 0:
                groups.append(cleaned.reset_index(drop=True))
        if not groups:
            raise ValueError("Dataset has no non-empty decision-date cross-sections")
        self._groups = groups

    def __len__(self) -> int:
        return len(self._groups)

    @property
    def dates(self) -> tuple[pd.Timestamp, ...]:
        """Chronologically ordered decision dates."""
        return tuple(pd.Timestamp(group[self.date_column].iloc[0]) for group in self._groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cross_section = self._groups[index]
        raw = torch.tensor(
            cross_section[self.factor_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        label_values = cross_section[self.return_columns].to_numpy(dtype=np.float32).T
        by_horizon = {
            horizon: label_values[position]
            for position, horizon in enumerate(self.label_horizons)
        }
        execution_array = None
        if self.execution_horizon is not None:
            execution_array = cross_section[self.execution_return_column].to_numpy(
                dtype=np.float32
            )
            by_horizon[int(self.execution_horizon)] = execution_array
        returns_array = np.stack([by_horizon[horizon] for horizon in self.horizons], axis=0)
        return_mask = np.isfinite(returns_array)
        forward_returns = torch.tensor(
            np.nan_to_num(returns_array, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=torch.float32,
        )
        benchmark: Tensor | None = None
        if all(column in cross_section.columns for column in self.benchmark_columns):
            values = cross_section[self.benchmark_columns].iloc[0].to_numpy(dtype=np.float32)
            benchmark = torch.tensor(values, dtype=torch.float32)
        industries = pd.to_numeric(cross_section[self.industry_column], errors="coerce")
        industry_ids = torch.tensor(industries.fillna(-1).to_numpy(np.int64), dtype=torch.long)
        return {
            "raw_factors": raw,
            "forward_returns": forward_returns,
            "return_mask": torch.tensor(return_mask, dtype=torch.bool),
            "asset_mask": torch.ones(len(cross_section), dtype=torch.bool),
            "asset_ids": [str(value) for value in cross_section[self.asset_column].tolist()],
            "industry_ids": industry_ids,
            "date": pd.Timestamp(cross_section[self.date_column].iloc[0]),
            "benchmark_returns": benchmark,
            "execution_1d_return": (
                None
                if execution_array is None
                else torch.tensor(
                    np.nan_to_num(execution_array, nan=0.0, posinf=0.0, neginf=0.0),
                    dtype=torch.float32,
                )
            ),
            "execution_return_mask": (
                None
                if execution_array is None
                else torch.tensor(np.isfinite(execution_array), dtype=torch.bool)
            ),
        }


def generate_synthetic_frame(
    num_dates: int = 40,
    num_stocks: int = 40,
    num_factors: int = 16,
    num_industries: int = 8,
    horizons: Sequence[int] = (3, 5, 10, 15, 20),
    seed: int = 42,
    active_probability: float = 0.90,
    signal_noise: float = 0.35,
) -> pd.DataFrame:
    """Create changing-universe data with a learnable nonlinear factor signal."""
    if not 0.0 < active_probability <= 1.0:
        raise ValueError("active_probability must be in (0, 1]")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=num_dates)
    industries = np.arange(num_stocks) % num_industries
    signal_weights = np.zeros(num_factors, dtype=np.float32)
    useful = min(5, num_factors)
    signal_weights[:useful] = np.asarray([0.7, -0.55, 0.4, 0.3, -0.2][:useful])
    rows: list[dict[str, Any]] = []
    for date_index, date in enumerate(dates):
        factors = rng.normal(size=(num_stocks, num_factors)).astype(np.float32)
        common = rng.normal(size=(num_industries, min(3, num_factors))).astype(np.float32)
        factors[:, : common.shape[1]] += common[industries]
        membership = rng.random(num_stocks) < active_probability
        membership[date_index % num_stocks] = True
        latent = factors @ signal_weights
        if num_factors >= 2:
            latent += 0.15 * factors[:, 0] * factors[:, 1]
        latent = (latent - latent.mean()) / (latent.std() + 1e-6)
        for stock in np.flatnonzero(membership):
            row: dict[str, Any] = {
                "date": date,
                "asset_id": f"S{stock:04d}",
                "industry_id": int(industries[stock]),
            }
            for factor_index in range(num_factors):
                value = float(factors[stock, factor_index])
                if rng.random() < 0.002:
                    value = float("nan")
                row[f"factor_{factor_index + 1:03d}"] = value
            for horizon in horizons:
                scale = np.sqrt(float(horizon) / max(horizons))
                row[f"forward_return_{horizon}"] = float(
                    0.01 * scale * latent[stock] + rng.normal(scale=0.01 * signal_noise)
                )
                row[f"benchmark_return_{horizon}"] = float(rng.normal(scale=0.001))
            rows.append(row)
    return pd.DataFrame(rows)
