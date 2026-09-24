import torch

from e2eai.models.deep_factor import MultiHorizonDeepFactor


def test_multi_horizon_heads_and_padding() -> None:
    module = MultiHorizonDeepFactor(4, [3, 5, 10, 15, 20])
    context = torch.randn(2, 6, 4)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    output = module(context, context, context, mask)
    assert output.shape == (2, 5, 6)
    assert (output.masked_select(~mask[:, None, :]) == 0).all()
    assert len(module.heads) == 5

