"""Validation metrics for decision-day weights and one-day execution returns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt

import numpy as np

from e2eai.evaluation.metrics import maximum_drawdown


def _simulate_daily_weight_path(
    target_weights: Sequence[Mapping[str, float]],
    execution_returns: Sequence[Mapping[str, float]],
    *,
    cost_bps: float,
    holding_threshold: float,
) -> dict[str, np.ndarray]:
    """Return per-date gross/net returns and drift-aware turnover arrays."""
    if len(target_weights) != len(execution_returns):
        raise ValueError("target_weights and execution_returns must have equal length")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    previous_posttrade: dict[str, float] | None = None
    net_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    effective: list[float] = []
    maximum_weights: list[float] = []
    selected_counts: list[float] = []
    hhi_values: list[float] = []

    for raw_target, raw_returns in zip(target_weights, execution_returns):
        target = {
            str(asset): max(0.0, float(weight))
            for asset, weight in raw_target.items()
            if np.isfinite(weight) and float(weight) > holding_threshold
        }
        total = sum(target.values())
        if total <= 0.0:
            continue
        target = {asset: weight / total for asset, weight in target.items()}
        returns = {
            str(asset): float(value)
            for asset, value in raw_returns.items()
            if np.isfinite(value)
        }
        observed = {asset: weight for asset, weight in target.items() if asset in returns}
        observed_total = sum(observed.values())
        if observed_total <= 0.0:
            continue
        observed = {asset: weight / observed_total for asset, weight in observed.items()}

        # One-way turnover is exactly one half of the L1 change from the
        # drifted pre-trade portfolio.  The cost formula then charges both
        # sides of the trade at the configured one-way bps rate.
        turnover = 0.0
        if previous_posttrade is not None:
            union = set(previous_posttrade) | set(target)
            turnover = 0.5 * sum(
                abs(target.get(asset, 0.0) - previous_posttrade.get(asset, 0.0))
                for asset in union
            )
        gross = sum(weight * returns[asset] for asset, weight in observed.items())
        cost = 2.0 * turnover * cost_bps / 10000.0
        net = gross - cost
        denominator = 1.0 + gross
        if denominator > 1e-12:
            previous_posttrade = {
                asset: weight * (1.0 + returns[asset]) / denominator
                for asset, weight in observed.items()
                if 1.0 + returns[asset] > 0.0
            }
            post_total = sum(previous_posttrade.values())
            if post_total > 0.0:
                previous_posttrade = {
                    asset: weight / post_total
                    for asset, weight in previous_posttrade.items()
                }
        else:
            previous_posttrade = dict(observed)

        weights_array = np.asarray(list(target.values()), dtype=np.float64)
        hhi = float(np.square(weights_array).sum())
        gross_returns.append(float(gross))
        net_returns.append(float(net))
        turnovers.append(float(turnover))
        hhi_values.append(hhi)
        effective.append(1.0 / max(hhi, 1e-12))
        maximum_weights.append(float(weights_array.max()))
        selected_counts.append(float((weights_array > holding_threshold).sum()))

    return {
        "net_returns": np.asarray(net_returns, dtype=np.float64),
        "gross_returns": np.asarray(gross_returns, dtype=np.float64),
        "turnovers": np.asarray(turnovers, dtype=np.float64),
        "effective": np.asarray(effective, dtype=np.float64),
        "maximum_weights": np.asarray(maximum_weights, dtype=np.float64),
        "selected_counts": np.asarray(selected_counts, dtype=np.float64),
        "hhi_values": np.asarray(hhi_values, dtype=np.float64),
    }


def daily_path_metrics(
    path: Mapping[str, np.ndarray],
    *,
    annual_trading_days: int = 252,
    benchmark_returns: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize a simulated path and optionally add CSI500 active metrics."""
    values = np.asarray(path["net_returns"], dtype=np.float64)
    if values.size == 0:
        result = {
            "validation_daily_return": 0.0,
            "validation_daily_mean_return": 0.0,
            "validation_daily_sharpe": 0.0,
            "validation_daily_mdd": 0.0,
            "validation_daily_turnover": 0.0,
            "validation_effective_n": 0.0,
            "validation_max_weight": 0.0,
            "validation_selected_stock_count": 0.0,
            "validation_hhi": 0.0,
            "validation_daily_gross_mean_return": 0.0,
        }
        return result
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    wealth = float(np.prod(1.0 + values))
    annualized = wealth ** (annual_trading_days / values.size) - 1.0 if wealth > 0.0 else -1.0
    result = {
        "validation_daily_return": float(annualized),
        "validation_daily_mean_return": mean,
        "validation_daily_sharpe": float(sqrt(annual_trading_days) * mean / std) if std > 1e-12 else 0.0,
        "validation_daily_mdd": maximum_drawdown(values),
        "validation_daily_turnover": float(np.mean(path["turnovers"])),
        "validation_effective_n": float(np.mean(path["effective"])),
        "validation_max_weight": float(np.max(path["maximum_weights"])),
        "validation_selected_stock_count": float(np.mean(path["selected_counts"])),
        "validation_hhi": float(np.mean(path["hhi_values"])),
        "validation_daily_gross_mean_return": float(np.mean(path["gross_returns"])),
    }
    if benchmark_returns is not None:
        benchmark = np.asarray(benchmark_returns, dtype=np.float64)
        if benchmark.size != values.size:
            raise ValueError("Strategy and benchmark daily paths must have equal length")
        active = values - benchmark
        active_std = float(active.std(ddof=0))
        benchmark_wealth = float(np.prod(1.0 + benchmark))
        benchmark_annual = (
            benchmark_wealth ** (annual_trading_days / benchmark.size) - 1.0
            if benchmark_wealth > 0.0 else -1.0
        )
        # "Annual excess return" is reported as the difference between the
        # strategy and benchmark annualized wealth returns.  The independently
        # compounded active path remains available as active_cumulative_return.
        result.update(
            validation_benchmark_annual_return=float(benchmark_annual),
            validation_active_annual_excess_return=float(annualized - benchmark_annual),
            validation_tracking_error=float(active_std * sqrt(annual_trading_days)),
            validation_information_ratio=float(sqrt(annual_trading_days) * active.mean() / active_std)
            if active_std > 1e-12 else 0.0,
            validation_active_max_drawdown=maximum_drawdown(active),
            validation_active_win_rate=float((active > 0.0).mean()),
            validation_active_cumulative_return=float(np.prod(1.0 + active) - 1.0),
        )
    return result


def evaluate_daily_weight_path(
    target_weights: Sequence[Mapping[str, float]],
    execution_returns: Sequence[Mapping[str, float]],
    *,
    cost_bps: float = 0.0,
    annual_trading_days: int = 252,
    holding_threshold: float = 1e-8,
) -> dict[str, float]:
    """Evaluate daily rebalancing with drift-aware one-way turnover.

    Entry weights are the decision at ``t`` executed at ``t+1``; the supplied
    execution return is ``VWAP[t+2]/VWAP[t+1]-1``.  The next rebalance compares
    its target with the prior portfolio after that one-day return drift.
    """
    path = _simulate_daily_weight_path(
        target_weights,
        execution_returns,
        cost_bps=cost_bps,
        holding_threshold=holding_threshold,
    )
    return daily_path_metrics(path, annual_trading_days=annual_trading_days)


__all__ = ["evaluate_daily_weight_path", "daily_path_metrics"]
