"""Build daily executed VWAP backtests for model, factor baseline, and index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e2eai.evaluation.daily_backtest import backtest_target_weights_daily
from e2eai.evaluation.metrics import maximum_drawdown
from scripts.build_multihorizon_labels import MarketArrays, _adjusted_vwap


def _market_prices(
    market: MarketArrays, dates: pd.DatetimeIndex, assets: list[str]
) -> pd.DataFrame:
    date_lookup = {date: index for index, date in enumerate(market.dates)}
    asset_lookup = {asset: index for index, asset in enumerate(market.assets)}
    first = date_lookup[pd.Timestamp(dates.min())]
    last_decision = date_lookup[pd.Timestamp(dates.max())]
    last = min(last_decision + 1, len(market.dates) - 1)
    rows = np.arange(first, last + 1, dtype=np.int64)
    columns = np.asarray([asset_lookup[asset] for asset in assets], dtype=np.int64)
    values = _adjusted_vwap(market, rows, columns)
    return pd.DataFrame(values, index=market.dates[rows], columns=assets).stack(dropna=True).rename(
        "price"
    ).rename_axis(["date", "asset_id"]).reset_index()


def _benchmark_targets(
    market: MarketArrays,
    dates: pd.DatetimeIndex,
    assets: list[str],
    field: str,
) -> pd.DataFrame:
    date_lookup = {date: index for index, date in enumerate(market.dates)}
    asset_lookup = {asset: index for index, asset in enumerate(market.assets)}
    rows = np.asarray([date_lookup[pd.Timestamp(date)] for date in dates], dtype=np.int64)
    columns = np.asarray([asset_lookup[asset] for asset in assets], dtype=np.int64)
    weights = market.matrix(field, rows, columns)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    return pd.DataFrame(weights, index=dates, columns=assets).stack().rename(
        "portfolio_weight"
    ).rename_axis(["date", "asset_id"]).reset_index()


def _active_metrics(portfolio: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, float]:
    joined = portfolio[["date", "portfolio_value"]].merge(
        benchmark[["date", "portfolio_value"]], on="date", suffixes=("_portfolio", "_benchmark")
    )
    active_wealth = joined["portfolio_value_portfolio"] / joined["portfolio_value_benchmark"]
    active_returns = active_wealth.pct_change().fillna(0.0).to_numpy(dtype=float)
    periods = max(len(joined), 1)
    return {
        "annualized_excess_return": float(active_wealth.iloc[-1] ** (252.0 / periods) - 1.0),
        "max_active_drawdown": maximum_drawdown(active_returns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--market-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-weight-field", default="zz500_weight")
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 15, 20])
    parser.add_argument("--cost-bps", type=float, default=0.0)
    args = parser.parse_args()

    model = pd.read_parquet(args.model_predictions)
    baseline = pd.read_parquet(args.baseline_predictions)
    dates = pd.DatetimeIndex(pd.to_datetime(model["date"].unique())).sort_values()
    assets = set(model["asset_id"].astype(str)) | set(baseline["asset_id"].astype(str))
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.market_h5, "r") as handle:
        market = MarketArrays(handle, "h5")
        date_lookup = {date: index for index, date in enumerate(market.dates)}
        decision_rows = np.asarray([date_lookup[pd.Timestamp(date)] for date in dates], dtype=np.int64)
        all_columns = np.arange(len(market.assets), dtype=np.int64)
        index_weights = market.matrix(args.benchmark_weight_field, decision_rows, all_columns)
        benchmark_assets = market.assets[np.any(np.isfinite(index_weights) & (index_weights > 0), axis=0)]
        assets = sorted(assets | set(benchmark_assets.astype(str)))
        prices = _market_prices(market, dates, assets)
        benchmark_targets = _benchmark_targets(
            market, dates, assets, args.benchmark_weight_field
        )

    all_metrics: dict[str, dict] = {}
    for horizon in args.horizons:
        horizon_dir = args.output / f"horizon_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        streams = {
            "model": model[model["horizon"] == horizon],
            "factor_equal_weight": baseline[baseline["horizon"] == horizon],
            "benchmark": benchmark_targets,
        }
        daily_frames: dict[str, pd.DataFrame] = {}
        horizon_metrics: dict[str, dict] = {}
        for name, targets in streams.items():
            daily, metrics = backtest_target_weights_daily(
                targets[["date", "asset_id", "portfolio_weight"]], prices, cost_bps=args.cost_bps
            )
            daily.to_parquet(horizon_dir / f"{name}_daily.parquet", index=False)
            daily_frames[name] = daily
            horizon_metrics[name] = metrics
        horizon_metrics["model"].update(_active_metrics(daily_frames["model"], daily_frames["benchmark"]))
        horizon_metrics["factor_equal_weight"].update(
            _active_metrics(daily_frames["factor_equal_weight"], daily_frames["benchmark"])
        )
        all_metrics[str(horizon)] = horizon_metrics
    metadata = {
        "entry": "decision date t targets are executed at adjusted VWAP on the next trading date t+1",
        "holdings": "positions drift with adjusted VWAP between daily target rebalances",
        "turnover": "one-way turnover is 0.5 times the L1 change from drifted pre-trade weights",
        "transaction_cost": f"{args.cost_bps} bps applied to gross turnover",
        "by_horizon": all_metrics,
    }
    (args.output / "daily_backtest_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
