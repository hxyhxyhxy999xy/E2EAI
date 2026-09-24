"""Leakage-safe cross-sectional preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def impute_factors(
    frame: pd.DataFrame,
    factor_columns: Sequence[str],
    strategy: str,
) -> pd.DataFrame:
    """Impute factor values without using another date's observations.

    ``cross_sectional_median`` replaces missing values with the same-date
    median and then zero if the entire cross-section is missing.
    ``zero_after_standardization`` performs same-date z-scoring with finite
    observations and replaces unresolved standardized values with zero.
    """
    result = frame.copy()
    result[list(factor_columns)] = result[list(factor_columns)].replace(
        [np.inf, -np.inf], np.nan
    )
    if strategy == "cross_sectional_median":
        medians = result[list(factor_columns)].median(axis=0, skipna=True)
        result[list(factor_columns)] = result[list(factor_columns)].fillna(medians).fillna(0.0)
    elif strategy == "zero_after_standardization":
        values = result[list(factor_columns)]
        means = values.mean(axis=0, skipna=True)
        standard_deviations = values.std(axis=0, skipna=True, ddof=0).replace(0.0, np.nan)
        result[list(factor_columns)] = ((values - means) / standard_deviations).fillna(0.0)
    else:
        raise ValueError(f"Unsupported missing-factor strategy: {strategy}")
    return result


def calculate_forward_returns(
    frame: pd.DataFrame,
    horizons: Sequence[int],
    date_column: str = "date",
    asset_column: str = "asset_id",
    price_column: str = "vwap",
    prefix: str = "forward_return_",
) -> pd.DataFrame:
    """Calculate paper-style labels ``P[t+k] / P[t+1] - 1`` by asset."""
    if price_column not in frame.columns:
        raise ValueError(f"Price column '{price_column}' is unavailable")
    result = frame.sort_values([asset_column, date_column]).copy()
    grouped = result.groupby(asset_column, sort=False)[price_column]
    next_price = grouped.shift(-1)
    for horizon in horizons:
        terminal = grouped.shift(-int(horizon))
        result[f"{prefix}{horizon}"] = terminal / next_price - 1.0
    return result
