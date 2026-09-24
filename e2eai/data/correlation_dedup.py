"""Deterministic train-only IC ranking with greedy correlation de-duplication.

Correlation is intentionally defined cross-sectionally *within each decision
date*, followed by a mean across permitted dates.  It is not a flattened panel
correlation and no function in this module infers a train period implicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from e2eai.data.train_only_selection import select_train_only_factors


@dataclass(frozen=True)
class CorrelationResult:
    """Mean daily cross-sectional Spearman correlation for one factor pair."""

    mean_correlation: float
    valid_date_count: int
    min_common_stock_count: int
    valid: bool


def mean_daily_cross_sectional_spearman(
    left: np.ndarray,
    right: np.ndarray,
    asset_mask: np.ndarray,
    *,
    min_common_stocks: int = 30,
    min_valid_dates: int = 100,
) -> CorrelationResult:
    """Return mean daily Spearman correlation on the pair's common valid stocks.

    ``left`` and ``right`` have shape ``[dates, assets]``.  A pair lacking the
    requested common-stock or common-date support is explicitly invalid; it is
    never silently interpreted as zero correlation.
    """
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    base = np.asarray(asset_mask, dtype=bool)
    if x.shape != y.shape or x.shape != base.shape or x.ndim != 2:
        raise ValueError("left, right, and asset_mask must share [dates, assets] shape")
    if min_common_stocks < 3 or min_valid_dates < 1:
        raise ValueError("min_common_stocks must be >= 3 and min_valid_dates must be positive")

    common = base & np.isfinite(x) & np.isfinite(y)
    counts = common.sum(axis=1)
    ranked_x = rankdata(np.where(common, x, np.nan), axis=1, method="average", nan_policy="omit")
    ranked_y = rankdata(np.where(common, y, np.nan), axis=1, method="average", nan_policy="omit")
    safe_count = np.maximum(counts.astype(np.float64), 1.0)
    mean_x = np.where(common, ranked_x, 0.0).sum(axis=1) / safe_count
    mean_y = np.where(common, ranked_y, 0.0).sum(axis=1) / safe_count
    centered_x = np.where(common, ranked_x - mean_x[:, None], 0.0)
    centered_y = np.where(common, ranked_y - mean_y[:, None], 0.0)
    denominator = np.sqrt((centered_x * centered_x).sum(axis=1) * (centered_y * centered_y).sum(axis=1))
    daily = np.divide(
        (centered_x * centered_y).sum(axis=1),
        denominator,
        out=np.full(x.shape[0], np.nan, dtype=np.float64),
        where=(counts >= min_common_stocks) & (denominator > 1e-12),
    )
    finite = daily[np.isfinite(daily)]
    return CorrelationResult(
        mean_correlation=float(finite.mean()) if finite.size >= min_valid_dates else float("nan"),
        valid_date_count=int(finite.size),
        min_common_stock_count=int(counts.min()) if counts.size else 0,
        valid=bool(finite.size >= min_valid_dates),
    )


CorrelationGetter = Callable[[str, str], CorrelationResult]


def greedy_ic_then_correlation_dedup(
    ranked_statistics: pd.DataFrame,
    correlation_getter: CorrelationGetter,
    *,
    top_k: int = 64,
    correlation_threshold: float = 0.80,
) -> tuple[pd.DataFrame, list[str]]:
    """Select factors in fixed IC order subject to ``abs(mean_corr) < threshold``.

    The supplied table must already be sorted by descending abs(mean RankIC),
    then ascending factor name.  Invalid pair correlations cause a conservative
    rejection, preserving the rule that missing correlation evidence cannot be
    treated as an uncorrelated pair.
    """
    required = {"rank", "factor_name", "train_mean_rankic", "abs_train_mean_rankic"}
    missing = required.difference(ranked_statistics.columns)
    if missing:
        raise ValueError(f"ranked_statistics missing columns: {sorted(missing)}")
    if not 1 <= top_k <= len(ranked_statistics):
        raise ValueError("top_k must be in [1, number of candidates]")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")

    result = ranked_statistics.copy().reset_index(drop=True)
    result["accepted"] = False
    result["selected_rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["max_abs_corr_to_selected_at_decision"] = np.nan
    result["most_correlated_selected_factor"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["signed_corr_with_most_correlated_factor"] = np.nan
    result["correlation_valid_date_count_at_decision"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["rejection_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")

    selected: list[str] = []
    target_reached = False
    for row_index, row in result.iterrows():
        name = str(row["factor_name"])
        if target_reached:
            result.at[row_index, "rejection_reason"] = "not_evaluated_after_target_reached"
            continue
        if not np.isfinite(float(row["train_mean_rankic"])):
            result.at[row_index, "rejection_reason"] = "invalid_train_rankic"
            continue
        if not selected:
            result.at[row_index, "accepted"] = True
            result.at[row_index, "selected_rank"] = 1
            selected.append(name)
            continue

        pair_results = [(other, correlation_getter(name, other)) for other in selected]
        invalid = [(other, value) for other, value in pair_results if not value.valid]
        if invalid:
            result.at[row_index, "rejection_reason"] = "invalid_correlation_to_selected"
            result.at[row_index, "correlation_valid_date_count_at_decision"] = min(
                value.valid_date_count for _, value in invalid
            )
            continue
        other, maximum = max(pair_results, key=lambda item: abs(item[1].mean_correlation))
        result.at[row_index, "max_abs_corr_to_selected_at_decision"] = abs(maximum.mean_correlation)
        result.at[row_index, "most_correlated_selected_factor"] = other
        result.at[row_index, "signed_corr_with_most_correlated_factor"] = maximum.mean_correlation
        result.at[row_index, "correlation_valid_date_count_at_decision"] = maximum.valid_date_count
        if abs(maximum.mean_correlation) < correlation_threshold:
            result.at[row_index, "accepted"] = True
            result.at[row_index, "selected_rank"] = len(selected) + 1
            selected.append(name)
            if len(selected) == top_k:
                target_reached = True
        else:
            result.at[row_index, "rejection_reason"] = "correlation_threshold"

    if len(selected) != top_k:
        raise RuntimeError(
            f"Correlation filter selected only {len(selected)} of requested {top_k} factors; threshold was not relaxed"
        )
    return result, selected


def train_only_ic_then_correlation_dedup(
    factor_names: list[str],
    daily_rankic: np.ndarray,
    factor_values: np.ndarray,
    asset_mask: np.ndarray,
    train_decision_mask: np.ndarray,
    *,
    top_k: int = 64,
    correlation_threshold: float = 0.80,
    min_common_stocks: int = 30,
    min_valid_dates: int = 100,
) -> tuple[pd.DataFrame, list[str]]:
    """Convenience implementation used by tests and small in-memory audits.

    All inputs are explicitly sliced to ``train_decision_mask`` before either
    ranking or correlations are calculated, making future perturbations inert.
    """
    values = np.asarray(factor_values, dtype=np.float64)
    mask = np.asarray(asset_mask, dtype=bool)
    train = np.asarray(train_decision_mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] != len(factor_names):
        raise ValueError("factor_values must have shape [dates, assets, factors]")
    if mask.shape != values.shape[:2] or train.shape != (values.shape[0],):
        raise ValueError("asset_mask/train_decision_mask do not align with factor_values")
    statistics = select_train_only_factors(factor_names, daily_rankic, train, top_k=top_k)
    train_values = values[train]
    train_mask = mask[train]
    index = {name: position for position, name in enumerate(factor_names)}
    cache: dict[tuple[str, str], CorrelationResult] = {}

    def get(left: str, right: str) -> CorrelationResult:
        key = tuple(sorted((left, right)))
        if key not in cache:
            cache[key] = mean_daily_cross_sectional_spearman(
                train_values[:, :, index[left]],
                train_values[:, :, index[right]],
                train_mask,
                min_common_stocks=min_common_stocks,
                min_valid_dates=min_valid_dates,
            )
        return cache[key]

    return greedy_ic_then_correlation_dedup(
        statistics,
        get,
        top_k=top_k,
        correlation_threshold=correlation_threshold,
    )
