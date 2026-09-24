import torch

from e2eai.models.gat import GraphAttentionLayer


def test_gat_masks_edges_padding_and_isolated_nodes() -> None:
    layer = GraphAttentionLayer(3, num_heads=2, dropout=0.0, add_self_loops=False)
    features = torch.randn(1, 4, 3)
    adjacency = torch.zeros(1, 4, 4, dtype=torch.bool)
    adjacency[0, 0, 1] = True
    mask = torch.tensor([[True, True, True, False]])
    output = layer(features, adjacency, mask, return_attention=True)
    assert output.features.shape == features.shape
    assert output.attention.shape == (1, 2, 4, 4)
    assert torch.isfinite(output.features).all()
    assert torch.isfinite(output.attention).all()
    assert (output.attention[0, :, 0, 1] > 0).all()
    assert (output.attention[0, :, 0, [0, 2, 3]] == 0).all()
    assert (output.features[0, 2:] == 0).all()

