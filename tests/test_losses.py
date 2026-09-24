import torch

from e2eai.training.cross_sectional import CrossSectionalRegressionSolver
from e2eai.training.losses import (
    E2EAILoss,
    approximation_objective,
    directional_factor_return_objective,
    directional_icir_objective,
    masked_cross_sectional_ic,
)


def test_all_losses_finite_and_attention_error_exact() -> None:
    torch.manual_seed(2)
    deep = torch.randn(4, 2, 7, requires_grad=True)
    returns = 0.2 * deep.detach() + 0.1 * torch.randn_like(deep)
    weights = torch.softmax(torch.randn(4, 2, 7), dim=-1)
    mask = torch.ones(4, 7, dtype=torch.bool)
    loss_fn = E2EAILoss(
        regression_solver=CrossSectionalRegressionSolver(lr=0.1, n_iter=2),
        theta=0.10,
    )
    result = loss_fn(deep, deep.detach().clone(), weights, returns, torch.ones(2), mask)
    assert all(torch.isfinite(value).all() for value in result.values() if value.is_floating_point())
    assert result["attention_estimate_loss"] == 0
    assert result["upper_bound_loss"] > 0
    result["total_loss"].backward()
    assert deep.grad is not None and torch.isfinite(deep.grad).all()


def test_directional_factor_return_sign_and_approximation_order() -> None:
    psi = torch.full((3, 2), 0.4)
    assert directional_factor_return_objective(psi, torch.ones(2)) < 0
    assert directional_factor_return_objective(psi, -torch.ones(2)) > 0
    deep = torch.randn(2, 2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    exact = approximation_objective(deep, deep, mask)
    shifted = approximation_objective(deep, deep + 1.0, mask)
    assert exact < shifted


def test_zero_variance_ic_and_icir_have_finite_backward() -> None:
    deep = torch.ones(4, 2, 7, requires_grad=True)
    returns = torch.arange(7, dtype=torch.float32).expand_as(deep)
    mask = torch.ones(4, 2, 7, dtype=torch.bool)
    ic, valid = masked_cross_sectional_ic(deep, returns, mask)
    assert not valid.any()
    stability = directional_icir_objective(
        deep,
        returns,
        torch.ones(2),
        mask,
        eps=1.0e-6,
        min_observations=3,
    )
    assert torch.isfinite(ic).all()
    assert torch.isfinite(stability.loss)
    (ic.sum() + stability.loss).backward()
    assert deep.grad is not None and torch.isfinite(deep.grad).all()


def test_zero_portfolio_weight_excludes_portfolio_value_but_score_backpropagates() -> None:
    torch.manual_seed(7)
    deep = torch.randn(3, 1, 6, requires_grad=True)
    returns = deep.detach() + 0.05 * torch.randn_like(deep)
    mask = torch.ones(3, 6, dtype=torch.bool)
    weights_a = torch.full((3, 1, 6), 1.0 / 6.0)
    weights_b = torch.zeros_like(weights_a)
    weights_b[..., 0] = 1.0
    loss_fn = E2EAILoss(
        lambda_portfolio=0.0,
        lambda_score=1.0,
        lambda_s=0.0,
        lambda_f=0.0,
        lambda_e=0.0,
        allocation_mode="topk_equal_weight",
    )
    result_a = loss_fn(
        deep,
        deep.detach(),
        weights_a,
        returns,
        torch.ones(1),
        mask,
        score_predictions=deep,
    )
    result_b = loss_fn(
        deep,
        deep.detach(),
        weights_b,
        returns,
        torch.ones(1),
        mask,
        score_predictions=deep,
    )
    assert result_a["portfolio_loss"] != result_b["portfolio_loss"]
    assert torch.allclose(result_a["total_loss"], result_b["total_loss"])
    assert result_a["weighted_portfolio_loss"] == 0
    result_a["total_loss"].backward()
    assert deep.grad is not None
    assert torch.isfinite(deep.grad).all()
    assert (deep.grad != 0).any()


def test_unit_portfolio_weight_preserves_previous_total_loss() -> None:
    torch.manual_seed(8)
    deep = torch.randn(3, 1, 6, requires_grad=True)
    returns = torch.randn_like(deep)
    weights = torch.softmax(torch.randn_like(deep), dim=-1)
    mask = torch.ones(3, 6, dtype=torch.bool)
    loss_fn = E2EAILoss(
        lambda_portfolio=1.0,
        lambda_score=0.0,
        lambda_s=0.0,
        lambda_f=0.0,
        lambda_e=0.0,
        allocation_mode="topk_equal_weight",
    )
    result = loss_fn(deep, deep.detach(), weights, returns, torch.ones(1), mask)
    assert torch.allclose(result["total_loss"], result["portfolio_loss"])
    assert torch.allclose(result["weighted_portfolio_loss"], result["portfolio_loss"])
