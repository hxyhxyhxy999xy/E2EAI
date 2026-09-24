import pandas as pd

from e2eai.evaluation.daily_backtest import backtest_target_weights_daily


def test_daily_backtest_executes_on_next_date_and_drift_positions():
    targets = pd.DataFrame(
        [
            {"date": "2020-01-02", "asset_id": "A", "portfolio_weight": 1.0},
            {"date": "2020-01-03", "asset_id": "B", "portfolio_weight": 1.0},
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2020-01-02", "asset_id": "A", "price": 10.0},
            {"date": "2020-01-02", "asset_id": "B", "price": 20.0},
            {"date": "2020-01-03", "asset_id": "A", "price": 11.0},
            {"date": "2020-01-03", "asset_id": "B", "price": 20.0},
            {"date": "2020-01-06", "asset_id": "A", "price": 12.0},
            {"date": "2020-01-06", "asset_id": "B", "price": 22.0},
        ]
    )
    daily, metrics = backtest_target_weights_daily(targets, prices, cost_bps=0.0)
    assert list(daily["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-03", "2020-01-06"]
    assert daily.iloc[0]["net_return"] == 0.0
    assert daily.iloc[1]["net_return"] > 0.0
    assert metrics["number_of_days"] == 2
