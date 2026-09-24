import torch

from e2eai.models.neutralization import (
    CascadedRelationalNeutralization,
    RelationalNeutralizationBlock,
)


def test_rnb_is_input_minus_gat() -> None:
    block = RelationalNeutralizationBlock(3, dropout=0.0)
    context = torch.randn(1, 4, 3)
    adjacency = torch.ones(1, 4, 4, dtype=torch.bool)
    mask = torch.ones(1, 4, dtype=torch.bool)
    expected = context - block.gat(context, adjacency, mask)
    assert torch.allclose(block(context, adjacency, mask), expected)
    diagnostic = block.forward_with_diagnostics(context, adjacency, mask)
    assert torch.allclose(diagnostic.neutral, expected)
    assert torch.allclose(diagnostic.gat_component, context - expected)


def test_neutralization_is_cascaded() -> None:
    module = CascadedRelationalNeutralization(3, dropout=0.0)
    context = torch.randn(1, 4, 3)
    graph = torch.ones(1, 4, 4, dtype=torch.bool)
    mask = torch.ones(1, 4, dtype=torch.bool)
    output = module(context, graph, graph, mask)
    expected_second = module.universe_block(output.industry_neutral, graph, mask)
    assert torch.allclose(output.universe_neutral, expected_second)
