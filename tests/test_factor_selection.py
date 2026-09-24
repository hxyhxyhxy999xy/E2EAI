import torch

from e2eai.models.factor_selection import FactorSelector


def test_factor_selection_shape_threshold_fallback_and_padding() -> None:
    selector = FactorSelector(5, hidden_dim=7, gamma_f=0.99, min_selected_factors=2)
    factors = torch.randn(2, 6, 5)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    output = selector(factors, mask)
    assert output.factor_attention.shape == (2, 5)
    assert torch.allclose(output.factor_attention.sum(-1), torch.ones(2))
    assert output.factor_selection_mask.dtype == torch.bool
    assert (output.factor_selection_mask.sum(-1) >= 2).all()
    assert (output.selected_factors.masked_select(~mask.unsqueeze(-1)) == 0).all()


def test_ste_gate_propagates_to_selector_mlp() -> None:
    selector = FactorSelector(4, hidden_dim=5, gamma_f=0.9, gate_mode="ste_hard")
    factors = torch.randn(3, 7, 4)
    output = selector(factors, torch.ones(3, 7, dtype=torch.bool))
    output.selected_factors.square().sum().backward()
    gradients = [parameter.grad for parameter in selector.mlp.parameters()]
    assert any(gradient is not None and (gradient != 0).any() for gradient in gradients)


def test_soft_train_and_eval_keep_the_same_continuous_gate() -> None:
    selector = FactorSelector(
        3,
        gamma_f=0.9,
        gate_mode_train="soft",
        gate_mode_eval="soft",
    )
    factors = torch.ones(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    selector.train()
    train_output = selector(factors, mask)
    selector.eval()
    eval_output = selector(factors, mask)
    assert torch.allclose(train_output.selected_factors, eval_output.selected_factors)
    assert ((eval_output.selected_factors > 0.0) & (eval_output.selected_factors < 1.0)).any()


def test_ste_hard_train_and_hard_eval_keep_paper_behavior() -> None:
    selector = FactorSelector(
        3,
        gamma_f=0.9,
        gate_mode_train="ste_hard",
        gate_mode_eval="hard",
    )
    factors = torch.ones(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    selector.train()
    train_output = selector(factors, mask)
    selector.eval()
    eval_output = selector(factors, mask)
    assert torch.equal(train_output.factor_selection_mask, eval_output.factor_selection_mask)
    assert set(torch.unique(eval_output.selected_factors).tolist()).issubset({0.0, 1.0})
