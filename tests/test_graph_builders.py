import torch

from e2eai.data.graphs import IndustryGraphBuilder, UniverseGraphBuilder


def test_dynamic_graph_modes_and_unknown_industry() -> None:
    industries = torch.tensor([[0, 0, 1, -1, -1]])
    valid = torch.tensor([[True, True, True, True, False]])
    industry = IndustryGraphBuilder(unknown_self_only=True)(industries, valid)
    assert industry[0, 0, 1]
    assert not industry[0, 0, 2]
    assert industry[0, 3, 3]
    assert not industry[0, 3, 4]

    full = UniverseGraphBuilder("all_valid")(industries, valid)
    assert full[0, :4, :4].all()
    assert not full[0, 4].any()
    cross = UniverseGraphBuilder("cross_industry_only")(industries, valid)
    assert cross[0, 0, 2]
    assert not cross[0, 0, 1]
    assert torch.diagonal(cross[0])[:4].all()

