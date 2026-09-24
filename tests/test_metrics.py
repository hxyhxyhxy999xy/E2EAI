import math

import torch

from e2eai.evaluation.backtest import backtest_multi_horizon
from e2eai.evaluation.metrics import compute_performance_metrics, turnover_from_asset_weights


def test_benchmark_aware_metrics_and_turnover_alignment() -> None:
    raw = compute_performance_metrics([0.01, -0.005, 0.02])
    assert raw["alpha"] is None
    active = compute_performance_metrics(
        [0.01, -0.005, 0.02], benchmark_returns=[0.0, 0.0, 0.0]
    )
    assert active["alpha"] is not None and math.isfinite(active["alpha"])
    turnover = turnover_from_asset_weights([{"A": 1.0}, {"B": 1.0}])
    assert turnover == {"one_way_turnover": 1.0, "gross_turnover": 2.0}


def test_backtest_labels_overlap_flag() -> None:
    weights = torch.full((4, 2, 3), 1 / 3)
    returns = torch.randn(4, 2, 3) * 0.01
    mask = torch.ones_like(returns, dtype=torch.bool)
    result = backtest_multi_horizon(weights, returns, mask, [3, 5])
    assert result["overlapping_labels"] is True
    assert result["statistical_independence_warning"]

