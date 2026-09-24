import torch

from e2eai.models.directional_attention import DirectionalFactorAttention


def test_directional_attention_shapes_sums_and_sign_effect() -> None:
    module = DirectionalFactorAttention(3, 2)
    for projection in module.projections:
        torch.nn.init.zeros_(projection.weight)
        torch.nn.init.zeros_(projection.bias)
    factors = torch.randn(2, 5, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    positive = module(factors, mask, torch.ones(2, 3))
    negative = module(factors, mask, -torch.ones(2, 3))
    assert positive.factor_attentions.shape == (2, 2, 5, 3)
    valid_sums = positive.factor_attentions.sum(-1).masked_select(mask[:, None, :])
    assert torch.allclose(valid_sums, torch.ones_like(valid_sums))
    assert torch.allclose(negative.deep_factor_approx, -positive.deep_factor_approx)
    positive.deep_factor_approx.square().sum().backward()
    assert factors.grad is not None

