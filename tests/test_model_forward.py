import inspect

import torch

from e2eai.models.e2eai import E2EAIModel


def test_full_model_shapes_padding_and_no_label_input(tiny_config, tiny_loaders) -> None:
    batch = next(iter(tiny_loaders.train))
    model = E2EAIModel(tiny_config.model)
    output = model(
        batch["raw_factors"],
        batch["asset_mask"],
        batch["industry_adjacency"],
        batch["universe_adjacency"],
        cap_mode="capped_simplex",
    )
    batch_size, stocks, factors = batch["raw_factors"].shape
    horizons = len(tiny_config.model.horizons)
    assert output.factor_attention.shape == (batch_size, factors)
    assert output.deep_factors.shape == (batch_size, horizons, stocks)
    assert output.factor_attentions.shape == (batch_size, horizons, stocks, factors)
    assert output.portfolio_weights.shape == (batch_size, horizons, stocks)
    assert torch.allclose(output.portfolio_weights.sum(-1), torch.ones(batch_size, horizons), atol=1e-5)
    invalid = ~batch["asset_mask"][:, None, :].expand_as(output.portfolio_weights)
    assert (output.portfolio_weights.masked_select(invalid) == 0).all()
    assert (output.portfolio_weights <= tiny_config.model.portfolio.theta + 1e-5).all()
    assert "forward_returns" not in inspect.signature(model.forward).parameters


def test_current_labels_cannot_change_preupdate_decision(tiny_config, tiny_loaders) -> None:
    batch = next(iter(tiny_loaders.train))
    model = E2EAIModel(tiny_config.model).eval()
    factor_direction, deep_direction = model.snapshot_directions()
    with torch.no_grad():
        before = model(
            batch["raw_factors"], batch["asset_mask"], batch["industry_adjacency"],
            batch["universe_adjacency"], factor_directions=factor_direction,
            deep_directions=deep_direction,
        )
    altered_labels = -1000.0 * batch["forward_returns"]
    model.update_direction_buffers(before, altered_labels, batch["asset_mask"], batch["return_mask"])
    assert torch.equal(before.deep_directions, deep_direction)
    assert torch.isfinite(before.portfolio_weights).all()

