from __future__ import annotations

import numpy as np
import pytest
import torch

from e2eai.evaluation.continuous_portfolio import (
    continuous_long_only_weights,
    cross_sectional_zscore,
    masked_softmax,
    continuous_long_only_weights_torch,
)
from e2eai.evaluation.daily_validation import _simulate_daily_weight_path


def test_masked_zscore_uses_valid_finite_values_only() -> None:
    z, mask, mean, std = cross_sectional_zscore([1.0, 3.0, 100.0, np.nan], [True, True, False, True])
    assert mask.tolist() == [True, True, False, False]
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)
    assert z.tolist() == pytest.approx([-1.0, 1.0, 0.0, 0.0])


def test_invalid_weights_are_zero_and_valid_weights_sum_to_one() -> None:
    weights, z, _, _ = continuous_long_only_weights([1.0, 2.0, np.nan], [True, True, False])
    assert weights[2] == 0.0
    assert np.all(weights[:2] > 0.0)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert z[2] == 0.0


def test_positive_affine_score_invariance() -> None:
    left, *_ = continuous_long_only_weights([1.0, 4.0, 2.0, 3.0])
    right, *_ = continuous_long_only_weights(np.asarray([1.0, 4.0, 2.0, 3.0]) * 10.0 + 3.0)
    assert np.allclose(left, right, rtol=0.0, atol=1e-12)


def test_temperature_one_is_the_default_and_no_cap_is_applied() -> None:
    weights_default, z, _, _ = continuous_long_only_weights([0.0, 1.0, 2.0])
    weights_explicit = masked_softmax(z, np.ones(3, dtype=bool), temperature=1.0)
    assert np.allclose(weights_default, weights_explicit)
    assert weights_default.max() < 1.0


def test_torch_mapping_is_autograd_compatible() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    weights, _, _, _ = continuous_long_only_weights_torch(scores)
    weights.sum().backward()
    assert scores.grad is not None
    assert torch.all(weights > 0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_drifted_turnover_and_7p5bp_cost_formula() -> None:
    path = _simulate_daily_weight_path(
        [{"A": 1.0}, {"B": 1.0}],
        [{"A": 0.0}, {"B": 0.0}],
        cost_bps=7.5,
        holding_threshold=1e-8,
    )
    assert path["turnovers"].tolist() == pytest.approx([0.0, 1.0])
    assert path["net_returns"].tolist() == pytest.approx([0.0, -0.0015])
