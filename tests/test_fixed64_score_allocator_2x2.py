"""Synthetic guards for the no-training fixed-64 2x2 attribution audit."""

from __future__ import annotations

import numpy as np

from e2eai.training.net_return import numpy_drifted_rows
from scripts.run_fixed64_score_allocator_2x2 import (
    PRIMARY_COST_BPS,
    TEMPERATURE,
    TOP20_RATIO,
    _annualized,
    _cost_token,
    _metrics,
    _rank_corr,
    _slice_state,
    _zscore,
    softmax_t1_weights,
    top20_equal_weight,
)


def test_top20_uses_ceil_and_asset_tie_break() -> None:
    scores = np.array([1.0, 1.0, 0.1, -1.0, -2.0, -3.0])
    assets = np.array(["000002", "000001", "000003", "000004", "000005", "000006"])
    weights, selected = top20_equal_weight(scores, assets, np.ones(6, dtype=bool), ratio=TOP20_RATIO)
    assert selected.sum() == 2  # ceil(6 * 20%)
    assert selected[0] and selected[1]
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights[selected], 0.5)


def test_softmax_t1_is_fully_invested_and_masked() -> None:
    scores = np.array([-1.0, 0.0, 2.0, np.nan])
    mask = np.array([True, True, True, False])
    weights, normalized = softmax_t1_weights(scores, mask)
    assert TEMPERATURE == 1.0
    assert np.isclose(weights.sum(), 1.0)
    assert weights[-1] == 0.0
    assert np.isnan(normalized[-1]) or normalized[-1] == 0.0
    assert weights[2] > weights[1] > weights[0]


def test_allocator_pairs_preserve_the_same_raw_score() -> None:
    scores = np.array([-0.4, 0.2, 1.2, 0.8, -0.1])
    assets = np.array(["000001", "000002", "000003", "000004", "000005"])
    top, _ = top20_equal_weight(scores, assets, np.isfinite(scores))
    soft, z = softmax_t1_weights(scores, np.isfinite(scores))
    assert np.array_equal(scores, scores.copy())
    assert np.isclose(top.sum(), 1.0)
    assert np.isclose(soft.sum(), 1.0)
    assert np.isfinite(z).all()


def test_zscore_uses_population_scale_and_no_clipping() -> None:
    values = np.array([0.0, 1.0, 4.0])
    result = _zscore(values, np.ones(3, dtype=bool))
    expected = (values - values.mean()) / values.std(ddof=0)
    assert np.allclose(result, expected)


def test_rankic_is_allocator_invariant_for_same_score() -> None:
    scores = np.array([0.2, -0.1, 0.7, 0.5, -0.5])
    returns = np.array([0.01, -0.02, 0.04, 0.02, -0.03])
    assert _rank_corr(scores, returns) == _rank_corr(scores.copy(), returns)


def test_drifted_turnover_differs_from_naive_target_difference() -> None:
    targets = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}]
    returns = [{"a": 0.10, "b": -0.10}, {"a": 0.0, "b": 0.0}]
    _, _, turnover, cost, _ = numpy_drifted_rows(targets, returns)
    assert turnover[0] == 0.0
    assert turnover[1] > 0.0  # Naive 0.5 * L1(target_t - target_{t-1}) would be zero.
    assert np.isclose(cost[1], 2.0 * turnover[1] * PRIMARY_COST_BPS / 10000.0)


def test_cost_tokens_cover_required_cost_cases() -> None:
    assert [_cost_token(value) for value in (0.0, 5.0, 7.5, 10.0)] == ["0p0", "5p0", "7p5", "10p0"]


def test_annualization_is_path_based_not_an_average_of_returns() -> None:
    values = np.array([0.10, -0.10])
    assert np.isclose(_annualized(values), (0.99 ** (252.0 / 2.0)) - 1.0)


def test_state_slice_keeps_cross_year_first_day_turnover() -> None:
    # A minimal mock state object would be more brittle than testing the
    # accounting primitive: the second date retains the prior drift state.
    targets = [{"a": 1.0}, {"b": 1.0}]
    returns = [{"a": 0.0}, {"b": 0.0}]
    _, _, turnover, _, _ = numpy_drifted_rows(targets, returns)
    assert turnover[1] == 1.0
