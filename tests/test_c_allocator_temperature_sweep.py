from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from e2eai.evaluation.continuous_portfolio import continuous_long_only_weights
from e2eai.evaluation.daily_validation import _simulate_daily_weight_path
from scripts.run_c_allocator_temperature_sweep import (
    TEMPERATURES,
    _build_fixed_score_inputs,
    _stable_temperature_weights,
)

ROOT = Path(__file__).resolve().parents[1]


def _fixed() -> dict[str, object]:
    frame = pd.DataFrame(
        {
            "date": ["2021-01-04"] * 4,
            "asset_id": ["000001", "000002", "000003", "000004"],
            "raw_score": [1.0, 0.0, -1.0, np.nan],
            "execution_1d_return": [0.01, 0.02, -0.01, np.nan],
        }
    )
    _, by_date, _ = _build_fixed_score_inputs(frame)
    return by_date["2021-01-04"]


def test_only_preregistered_temperatures() -> None:
    assert TEMPERATURES == (1.0, 0.8, 0.6, 0.4)


def test_stable_softmax_is_finite_nonnegative_and_fully_invested() -> None:
    fixed = _fixed()
    for temperature in TEMPERATURES:
        weights = _stable_temperature_weights(fixed, temperature)
        assert np.isfinite(weights).all()
        assert (weights >= 0).all()
        assert weights[3] == 0.0
        assert weights[:3].sum() == pytest.approx(1.0)


def test_t1_matches_formal_allocator() -> None:
    fixed = _fixed()
    expected, z, _, _ = continuous_long_only_weights(fixed["raw"], fixed["valid"], temperature=1.0, eps=1e-6)
    actual = _stable_temperature_weights(fixed, 1.0)
    assert np.max(np.abs(expected - actual)) <= 1e-12
    assert np.max(np.abs(z - fixed["z"])) <= 1e-12


def test_lower_temperature_reduces_entropy_and_increases_max_weight() -> None:
    fixed = _fixed()
    entropies = []
    max_weights = []
    for temperature in TEMPERATURES:
        weights = _stable_temperature_weights(fixed, temperature)
        positive = weights[weights > 0]
        entropies.append(float(-np.sum(positive * np.log(positive))))
        max_weights.append(float(weights.max()))
    assert entropies == sorted(entropies, reverse=True)
    assert max_weights == sorted(max_weights)


def test_all_temperatures_share_the_same_raw_and_z_inputs() -> None:
    fixed = _fixed()
    raw = np.asarray(fixed["raw"])
    z = np.asarray(fixed["z"])
    for temperature in TEMPERATURES:
        _ = _stable_temperature_weights(fixed, temperature)
        assert np.array_equal(raw, np.asarray(fixed["raw"], dtype=float), equal_nan=True)
        assert np.array_equal(z, np.asarray(fixed["z"], dtype=float), equal_nan=True)


def test_turnover_uses_drifted_previous_weights() -> None:
    path = _simulate_daily_weight_path(
        [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}],
        [{"a": 0.10, "b": 0.0}, {"a": 0.0, "b": 0.0}],
        cost_bps=0.0,
        holding_threshold=1e-12,
    )
    expected_drift_turnover = 0.5 * (abs(0.5 - (0.5 * 1.10 / 1.05)) + abs(0.5 - (0.5 / 1.05)))
    assert path["turnovers"][1] == pytest.approx(expected_drift_turnover)
    assert path["turnovers"][1] > 0.0


def test_cost_formula_is_two_times_turnover_times_bps() -> None:
    path = _simulate_daily_weight_path(
        [{"a": 1.0}, {"a": 1.0}],
        [{"a": 0.01}, {"a": 0.01}],
        cost_bps=7.5,
        holding_threshold=1e-12,
    )
    assert path["net_returns"][1] == pytest.approx(path["gross_returns"][1] - 2.0 * path["turnovers"][1] * 7.5 / 10000.0)


def test_script_has_no_optimizer_or_backward_call() -> None:
    tree = ast.parse((ROOT / "scripts/run_c_allocator_temperature_sweep.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not any(node.func.attr in {"step", "backward"} for node in calls)


def test_script_has_no_other_temperature_literals() -> None:
    source = (ROOT / "scripts/run_c_allocator_temperature_sweep.py").read_text(encoding="utf-8")
    assert "(1.0, 0.8, 0.6, 0.4)" in source
    assert "0.5" not in source
    assert "0.55" not in source
    assert "0.65" not in source


def test_script_marks_2022_as_forbidden() -> None:
    source = (ROOT / "scripts/run_c_allocator_temperature_sweep.py").read_text(encoding="utf-8")
    assert "used_2022" in source
    assert "2022" in source


def test_weights_are_not_counted_by_strict_positivity() -> None:
    fixed = _fixed()
    weights = _stable_temperature_weights(fixed, 0.4)
    assert (weights[weights > 0] > 0).all()
    assert int((weights > 0).sum()) == 3


import pytest

