"""Explicit, benchmark-aware portfolio and factor metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _finite_array(values: Sequence[float] | np.ndarray | Tensor) -> np.ndarray:
    if isinstance(values, Tensor):
        array = values.detach().float().cpu().numpy()
    else:
        array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def maximum_drawdown(returns: Sequence[float] | np.ndarray | Tensor) -> float:
    """Maximum peak-to-trough loss on a compounded wealth curve."""
    values = _finite_array(returns)
    if values.size == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - wealth / np.maximum(peaks, 1e-12)
    return float(np.max(drawdowns))


def compute_performance_metrics(
    portfolio_returns: Sequence[float] | np.ndarray | Tensor,
    *,
    benchmark_returns: Sequence[float] | np.ndarray | Tensor | None = None,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
) -> dict[str, float | None]:
    """Compute return, Sharpe, drawdown, and benchmark-aware active metrics.

    ``alpha`` is ``None`` when no benchmark is supplied; raw return is never
    silently relabeled as alpha.
    """
    returns = _finite_array(portfolio_returns)
    if returns.size == 0:
        return {
            "annualized_return": float("nan"),
            "alpha": None,
            "information_ratio": None,
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "max_active_drawdown": None,
        }
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    periodic_rf = risk_free_rate / annualization
    sharpe = sqrt(annualization) * (mean - periodic_rf) / std if std > 1e-12 else 0.0
    result: dict[str, float | None] = {
        "annualized_return": mean * annualization,
        "alpha": None,
        "information_ratio": None,
        "sharpe": float(sharpe),
        "max_drawdown": maximum_drawdown(returns),
        "max_active_drawdown": None,
    }
    if benchmark_returns is not None:
        benchmark = _finite_array(benchmark_returns)
        if benchmark.size != returns.size:
            raise ValueError("Portfolio and benchmark return lengths must match after filtering")
        active = returns - benchmark
        active_std = float(np.std(active, ddof=0))
        result["alpha"] = float(np.mean(active) * annualization)
        result["information_ratio"] = (
            float(sqrt(annualization) * np.mean(active) / active_std)
            if active_std > 1e-12
            else 0.0
        )
        result["max_active_drawdown"] = maximum_drawdown(active)
    return result


def portfolio_concentration_metrics(
    weights: Tensor,
    holding_threshold: float = 1e-8,
) -> dict[str, float]:
    """Summarize holdings, top weight, Herfindahl index, and effective holdings."""
    detached = weights.detach().float()
    concentration = detached.square().sum(dim=-1)
    return {
        "average_holdings": float((detached > holding_threshold).sum(-1).float().mean()),
        "average_top_weight": float(detached.max(-1).values.mean()),
        "average_concentration": float(concentration.mean()),
        "effective_holdings": float((1.0 / concentration.clamp_min(1e-12)).mean()),
    }


def turnover_from_asset_weights(
    dated_weights: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Compute turnover after aligning stable asset IDs across adjacent dates."""
    if len(dated_weights) < 2:
        return {"one_way_turnover": 0.0, "gross_turnover": 0.0}
    gross_values: list[float] = []
    for previous, current in zip(dated_weights[:-1], dated_weights[1:]):
        union = set(previous) | set(current)
        gross_values.append(
            sum(abs(float(current.get(asset, 0.0)) - float(previous.get(asset, 0.0))) for asset in union)
        )
    gross = float(np.mean(gross_values))
    return {"one_way_turnover": 0.5 * gross, "gross_turnover": gross}


def factor_series_metrics(
    ic: Tensor,
    factor_returns: Tensor | None = None,
) -> dict[str, Any]:
    """Return per-horizon IC/ICIR and optional factor-return t statistics."""
    values = ic.detach().float()
    mean_ic = values.mean(dim=0)
    std_ic = values.std(dim=0, unbiased=False)
    result: dict[str, Any] = {
        "mean_ic": mean_ic.cpu().tolist(),
        "icir": (mean_ic / std_ic.clamp_min(1e-12)).cpu().tolist(),
    }
    if factor_returns is not None:
        returns = factor_returns.detach().float()
        count = returns.shape[0]
        mean_factor = returns.mean(dim=0)
        std_factor = returns.std(dim=0, unbiased=False)
        t_stat = mean_factor / (std_factor / sqrt(max(count, 1))).clamp_min(1e-12)
        result["factor_return_mean"] = mean_factor.cpu().tolist()
        result["factor_return_t_stat"] = t_stat.cpu().tolist() if count >= 2 else None
    return result

