import io

import torch

from e2eai.models.directional_buffer import DirectionalBuffer, masked_cross_sectional_ic


def test_masked_ic_and_buffer_update_serialization() -> None:
    x = torch.tensor([[[-2.0, -1.0, 0.0, 1.0, 2.0]]], requires_grad=True)
    y = 2.0 * x + 1.0
    ic = masked_cross_sectional_ic(x, y, torch.ones_like(x, dtype=torch.bool))
    assert torch.allclose(ic, torch.ones_like(ic), atol=1e-6)
    ic.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    buffer = DirectionalBuffer(1, 2, normalization="none", min_stocks=3)
    factors = torch.stack((x.detach()[0, 0], -x.detach()[0, 0]), dim=-1).unsqueeze(0)
    deep = x.detach()
    returns = x.detach()
    update = buffer.update(factors, deep, returns, torch.ones(1, 5, dtype=torch.bool))
    assert update["factor_direction"].tolist() == [[1.0, -1.0]]
    assert buffer.deep_directions().tolist() == [1.0]
    assert all(not parameter.requires_grad for parameter in buffer.parameters())
    memory = io.BytesIO()
    torch.save(buffer.state_dict(), memory)
    memory.seek(0)
    restored = DirectionalBuffer(1, 2, normalization="none")
    restored.load_state_dict(torch.load(memory, weights_only=True))
    assert torch.equal(restored.original_factor_ic_sum, buffer.original_factor_ic_sum)


def test_cross_factor_zscore_direction_and_reset() -> None:
    buffer = DirectionalBuffer(1, 3, normalization="cross_factor_zscore")
    buffer.original_factor_ic_sum.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    assert buffer.factor_directions().tolist() == [[-1.0, 1.0, 1.0]]
    buffer.reset()
    assert torch.allclose(buffer.original_factor_ic_sum, torch.full((1, 3), buffer.eps))

