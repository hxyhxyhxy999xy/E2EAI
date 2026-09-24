import torch

from e2eai.training.cross_sectional import CrossSectionalRegressionSolver


def test_local_solver_recovers_positive_slope_and_meta_gradient() -> None:
    torch.manual_seed(3)
    factor = torch.randn(2, 2, 80, requires_grad=True)
    returns = 0.01 + 0.4 * factor.detach() + 0.005 * torch.randn_like(factor)
    solver = CrossSectionalRegressionSolver(lr=0.25, n_iter=80, differentiable=True)
    result = solver(factor, returns)
    assert torch.allclose(result.psi.mean(), torch.tensor(0.4), atol=0.03)
    (-result.psi.mean()).backward()
    assert factor.grad is not None and torch.isfinite(factor.grad).all()


def test_detached_inner_loop_returns_detached_coefficients() -> None:
    factor = torch.randn(1, 1, 10, requires_grad=True)
    returns = 0.2 * factor.detach()
    result = CrossSectionalRegressionSolver(n_iter=2, differentiable=False)(factor, returns)
    assert not result.alpha.requires_grad
    assert not result.psi.requires_grad

