from __future__ import annotations

import pytest

from e2eai.config import load_config
from e2eai.evaluation.daily_validation import evaluate_daily_weight_path


def test_daily_core_is_execution_only_h2() -> None:
    config = load_config("configs/daily_core.yaml")
    assert config.model.horizons == [2]
    assert config.model.execution_horizon == 2
    assert config.data.panel_return_horizons == []
    assert config.data.label_exit_offsets == [2]
    assert config.training.optimization_horizons == [2]
    assert config.training.validation_target == "validation_daily_sharpe_7_5bps"
    assert config.evaluation.cost_scenarios_bps == [0.0, 5.0, 7.5, 10.0]


def test_one_way_turnover_and_seven_point_five_bps_cost() -> None:
    targets = [{"A": 1.0}, {"B": 1.0}]
    returns = [{"A": 0.0}, {"B": 0.0}]
    metrics = evaluate_daily_weight_path(targets, returns, cost_bps=7.5)
    # First observation has no turnover; the second has one-way turnover 1.0,
    # hence cost = 2 * 1.0 * 7.5 / 10000 = 0.0015 for that date.
    assert metrics["validation_daily_turnover"] == pytest.approx(0.5)
    assert metrics["validation_daily_mean_return"] == pytest.approx(-0.00075)
    assert metrics["validation_daily_gross_mean_return"] == pytest.approx(0.0)


def test_daily_core_does_not_require_multiday_label_file() -> None:
    config = load_config("configs/daily_core.yaml")
    # The configured return and execution paths are the same audited 1-D
    # execution array; no 3/5/10/15/20 label path is present in this config.
    assert config.data.return_path.endswith("execution_1d_return_verified_2018_2021.npy")
    assert config.data.execution_return_path == config.data.return_path
    assert not any(h in str(config.data.return_path) for h in ("3_5_10_15_20", "forward_returns"))
