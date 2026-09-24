import torch

from e2eai.models.normalization import MaskedCrossSectionalNorm


def test_cross_sectional_normalization_ignores_padding() -> None:
    values = torch.tensor(
        [[[1.0, 4.0], [2.0, 6.0], [3.0, 8.0], [1000.0, -1000.0]]]
    )
    mask = torch.tensor([[True, True, True, False]])
    normalized = MaskedCrossSectionalNorm(eps=1e-8)(values, mask)
    valid = normalized[0, :3]
    assert torch.allclose(valid.mean(0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(valid.var(0, unbiased=False), torch.ones(2), atol=1e-5)
    assert (normalized[0, 3] == 0).all()


def test_zero_variance_and_nan_are_safe() -> None:
    values = torch.tensor([[[1.0, float("nan")], [1.0, 2.0]]])
    output = MaskedCrossSectionalNorm()(values, torch.ones(1, 2, dtype=torch.bool))
    assert torch.isfinite(output).all()

