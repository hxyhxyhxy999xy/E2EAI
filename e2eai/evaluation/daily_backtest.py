"""Executed daily backtest from target weights and adjusted VWAP prices.

The model evaluation files contain overlapping forward labels.  This module is
deliberately separate: it turns each decision-date target portfolio into an
executable position at the next trading day's VWAP, drifts positions between
rebalances, and charges optional transaction costs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from e2eai.evaluation.metrics import maximum_drawdown


def _normalise_weights(frame: pd.DataFrame) -> dict[str, float]:
    values = pd.to_numeric(frame["portfolio_weight"], errors="coerce").fillna(0.0)
    values = values.clip(lower=0.0)
    total = float(values.sum())
    if total <= 0.0:
        return {}
    return {
        str(asset): float(weight / total)
        for asset, weight in zip(frame["asset_id"], values / total)
        if float(weight) > 0.0
    }


def _next_trading_date(decision_date: pd.Timestamp, price_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    positions = price_dates.searchsorted(decision_date, side="right")
    return None if positions >= len(price_dates) else pd.Timestamp(price_dates[positions])


def backtest_target_weights_daily(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a daily VWAP execution backtest.

    Parameters
    ----------
    target_weights:
        Long form columns ``date``, ``asset_id`` and ``portfolio_weight``.
        A row on decision date *t* is executed at the next available price
        date (t+1), matching the paper's VWAP entry convention.
    prices:
        Long form columns ``date``, ``asset_id`` and ``price``.  Prices should
        be adjusted VWAPs and must be positive.
    cost_bps:
        Proportional cost applied to gross turnover; one-way turnover is
        0.5 * L1 target-weight change.
    """
    required_weights = {"date", "asset_id", "portfolio_weight"}
    required_prices = {"date", "asset_id", "price"}
    if not required_weights.issubset(target_weights.columns):
        raise ValueError(f"target_weights must contain {sorted(required_weights)}")
    if not required_prices.issubset(prices.columns):
        raise ValueError(f"prices must contain {sorted(required_prices)}")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")

    weights = target_weights.copy()
    weights["date"] = pd.to_datetime(weights["date"])
    weights = weights.sort_values(["date", "asset_id"])
    price_frame = prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame["price"] = pd.to_numeric(price_frame["price"], errors="coerce")
    price_frame = price_frame[price_frame["price"] > 0].sort_values(["date", "asset_id"])
    if weights.empty or price_frame.empty:
        raise ValueError("Both target_weights and prices must be non-empty")

    price_dates = pd.DatetimeIndex(price_frame["date"].drop_duplicates().sort_values())
    raw_price_table = price_frame.pivot_table(index="date", columns="asset_id", values="price", aggfunc="last")
    price_table = raw_price_table.reindex(price_dates).ffill()
    decisions: dict[pd.Timestamp, dict[str, float]] = {}
    for date, group in weights.groupby("date", sort=True):
        execution_date = _next_trading_date(pd.Timestamp(date), price_dates)
        if execution_date is not None:
            decisions[execution_date] = _normalise_weights(group)
    if not decisions:
        raise ValueError("No decision date has a following price date")

    first_execution = min(decisions)
    valuation_dates = price_dates[price_dates >= first_execution]
    holdings: dict[str, float] = {}
    value = 1.0
    rows: list[dict[str, float | pd.Timestamp]] = []
    previous_prices: pd.Series | None = None
    for date in valuation_dates:
        current_prices = price_table.loc[date]
        gross_return = 0.0
        turnover = 0.0
        transaction_cost = 0.0
        if previous_prices is not None and holdings:
            drifted = {
                asset: weight * float(current_prices.get(asset, np.nan)) / float(previous_prices.get(asset, np.nan))
                for asset, weight in holdings.items()
                if np.isfinite(current_prices.get(asset, np.nan))
                and np.isfinite(previous_prices.get(asset, np.nan))
                and previous_prices.get(asset, np.nan) > 0
            }
            drift_total = sum(drifted.values())
            gross_return = float(drift_total - 1.0) if drift_total > 0 else 0.0
            value *= 1.0 + gross_return
            holdings = (
                {asset: weight / drift_total for asset, weight in drifted.items()}
                if drift_total > 0 else {}
            )
        if date in decisions:
            observed_prices = raw_price_table.loc[date]
            target = {
                asset: weight for asset, weight in decisions[date].items()
                if np.isfinite(observed_prices.get(asset, np.nan))
                and observed_prices.get(asset, np.nan) > 0
            }
            available_total = sum(target.values())
            target = (
                {asset: weight / available_total for asset, weight in target.items()}
                if available_total > 0 else holdings
            )
            if holdings:
                pretrade = holdings
                union = set(pretrade) | set(target)
                turnover = 0.5 * sum(abs(target.get(asset, 0.0) - pretrade.get(asset, 0.0)) for asset in union)
            transaction_cost = (2.0 * turnover) * cost_bps / 10000.0
            value *= 1.0 - transaction_cost
            holdings = target
        rows.append({
            "date": date,
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": value / (rows[-1]["portfolio_value"] if rows else 1.0) - 1.0,
            "portfolio_value": value,
        })
        previous_prices = current_prices

    daily = pd.DataFrame(rows)
    wealth = daily["portfolio_value"].to_numpy(dtype=float)
    periods = max(len(daily), 1)
    metrics = {
        "annualized_return": float(wealth[-1] ** (252.0 / periods) - 1.0),
        "max_drawdown": maximum_drawdown(daily["net_return"].to_numpy(dtype=float)),
        "average_one_way_turnover": float(daily["turnover"].mean()),
        "total_one_way_turnover": float(daily["turnover"].sum()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        "number_of_days": int(len(daily)),
    }
    return daily, metrics


def save_daily_backtest(
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    destination: str | Path,
    *,
    cost_bps: float = 0.0,
) -> dict[str, float]:
    """Run and save ``daily_returns.parquet`` and ``daily_metrics.json``."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    daily, metrics = backtest_target_weights_daily(target_weights, prices, cost_bps=cost_bps)
    daily.to_parquet(destination / "daily_returns.parquet", index=False)
    pd.Series(metrics, dtype=object).to_json(destination / "daily_metrics.json", force_ascii=False, indent=2)
    return metrics
