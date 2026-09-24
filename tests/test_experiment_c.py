from __future__ import annotations

import numpy as np
import torch
import pytest

from e2eai.evaluation.continuous_portfolio import continuous_long_only_weights_torch_batch


def _gross_loss(scores: torch.Tensor, returns: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights, _, _, _ = continuous_long_only_weights_torch_batch(scores, valid)
    observed = valid & torch.isfinite(returns)
    observed_weight = torch.where(observed, weights, torch.zeros_like(weights)).sum(dim=-1)
    gross = torch.where(
        observed_weight > 1e-6,
        torch.where(observed, weights * returns, torch.zeros_like(returns)).sum(dim=-1)
        / observed_weight.clamp_min(1e-6),
        torch.zeros_like(observed_weight),
    )
    return -gross.mean()


def test_gross_return_has_finite_nonzero_mlp_gradient() -> None:
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]], requires_grad=True)
    returns = torch.tensor([[0.01, -0.02, 0.03, 0.01]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    loss = _gross_loss(scores, returns, valid)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert float(scores.grad.norm()) > 0.0


def test_batched_allocator_masks_invalid_and_fully_invests() -> None:
    scores = torch.tensor([[1.0, 2.0, 99.0], [0.0, 1.0, 2.0]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    weights, normalized, _, _ = continuous_long_only_weights_torch_batch(scores, valid)
    assert weights[0, 2] == 0.0
    assert torch.all(weights >= 0.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    assert torch.isfinite(normalized).all()


def test_gross_loss_excludes_invalid_execution_returns() -> None:
    scores = torch.tensor([[0.0, 1.0]], requires_grad=True)
    returns = torch.tensor([[0.10, np.nan]])
    valid = torch.tensor([[True, True]])
    loss = _gross_loss(scores, returns, valid)
    # Only the observed first asset remains after the validation mask.
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(-0.10)
