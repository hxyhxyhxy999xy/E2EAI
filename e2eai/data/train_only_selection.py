"""Deterministic train-only factor selection from daily RankIC observations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_train_only_factors(
    factor_names: list[str] | tuple[str, ...],
    daily_rankic: np.ndarray,
    train_decision_mask: np.ndarray,
    *,
    top_k: int = 64,
) -> pd.DataFrame:
    """Summarize and select factors using only explicitly permitted dates.

    ``train_decision_mask`` is required instead of an implicit date boundary so
    callers cannot accidentally incorporate validation or test observations.
    Ties are broken by factor name, independent of process ordering or seed.
    """
    names = [str(name) for name in factor_names]
    values = np.asarray(daily_rankic, dtype=np.float64)
    mask = np.asarray(train_decision_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("daily_rankic must have shape [dates, factors]")
    if mask.shape != (values.shape[0],):
        raise ValueError("train_decision_mask must have one value per decision date")
    if not 1 <= top_k <= len(names):
        raise ValueError("top_k must be in [1, number of factors]")
    selected = values[mask]
    valid_count = np.isfinite(selected).sum(axis=0)
    means = np.full(len(names), np.nan, dtype=np.float64)
    stds = np.full(len(names), np.nan, dtype=np.float64)
    positive = np.full(len(names), np.nan, dtype=np.float64)
    for index in range(len(names)):
        factor_values = selected[:, index]
        finite = factor_values[np.isfinite(factor_values)]
        if finite.size:
            means[index] = float(finite.mean())
            stds[index] = float(finite.std(ddof=0))
            positive[index] = float((finite > 0.0).mean())
    icir = np.divide(
        means,
        stds,
        out=np.zeros_like(means),
        where=np.isfinite(stds) & (stds > 1e-12),
    )
    summary = pd.DataFrame(
        {
            "factor_name": names,
            "train_mean_rankic": means,
            "train_rankic_std": stds,
            "train_icir": icir,
            "train_positive_ic_ratio": positive,
            "valid_date_count": valid_count.astype(int),
        }
    )
    summary["abs_train_mean_rankic"] = summary["train_mean_rankic"].abs()
    summary = summary.sort_values(
        ["abs_train_mean_rankic", "factor_name"],
        ascending=[False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1, dtype=int))
    summary["selected"] = summary["rank"] <= int(top_k)
    summary["train_only_direction"] = np.where(
        summary["train_mean_rankic"] < 0.0,
        -1,
        1,
    ).astype(int)
    return summary
