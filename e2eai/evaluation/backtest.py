"""Label-based overlapping and non-overlapping multi-horizon evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from e2eai.evaluation.metrics import (
    compute_performance_metrics,
    portfolio_concentration_metrics,
)


def portfolio_returns_from_labels(
    weights: Tensor,
    forward_returns: Tensor,
    return_mask: Tensor,
) -> Tensor:
    """Compute ``[T,K]`` label returns without treating missing labels as observed."""
    if weights.shape != forward_returns.shape or return_mask.shape != weights.shape:
        raise ValueError("weights, forward_returns, and return_mask must share [T,K,N]")
    valid_values = torch.where(return_mask.bool(), forward_returns, torch.zeros_like(forward_returns))
    return (weights * valid_values).sum(dim=-1)


def backtest_multi_horizon(
    weights: Tensor,
    forward_returns: Tensor,
    return_mask: Tensor,
    horizons: Sequence[int],
    *,
    benchmark_returns: Tensor | None = None,
    mode: str = "overlapping_label",
    annual_trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Evaluate every horizon and explicitly identify overlapping-label results."""
    if mode not in {"overlapping_label", "non_overlapping"}:
        raise ValueError(f"Unknown backtest mode: {mode}")
    returns = portfolio_returns_from_labels(weights, forward_returns, return_mask)
    if returns.shape[1] != len(horizons):
        raise ValueError("Number of horizons does not match the K tensor dimension")
    by_horizon: dict[str, dict[str, float | None]] = {}
    for index, horizon in enumerate(horizons):
        stride = 1 if mode == "overlapping_label" else int(horizon)
        series = returns[::stride, index]
        benchmark = benchmark_returns[::stride, index] if benchmark_returns is not None else None
        annualization = annual_trading_days / (1 if mode == "overlapping_label" else horizon)
        by_horizon[str(horizon)] = compute_performance_metrics(
            series,
            benchmark_returns=benchmark,
            annualization=annualization,
            risk_free_rate=risk_free_rate,
        )
    return {
        "mode": mode,
        "overlapping_labels": mode == "overlapping_label",
        "statistical_independence_warning": (
            "Daily k-day labels overlap and are not statistically independent."
            if mode == "overlapping_label"
            else None
        ),
        "by_horizon": by_horizon,
        "concentration": portfolio_concentration_metrics(weights),
    }

