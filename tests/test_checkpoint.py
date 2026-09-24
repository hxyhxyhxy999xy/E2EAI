from pathlib import Path

import torch

from e2eai.models.e2eai import E2EAIModel
from e2eai.training.trainer import E2EAITrainer


def _strict_output(model, batch):
    model.eval()
    with torch.no_grad():
        return model(
            batch["raw_factors"], batch["asset_mask"], batch["industry_adjacency"],
            batch["universe_adjacency"], cap_mode="capped_simplex"
        ).portfolio_weights


def test_checkpoint_round_trip_reproduces_eval(tiny_config, tiny_loaders) -> None:
    batch = next(iter(tiny_loaders.train))
    model = E2EAIModel(tiny_config.model)
    trainer = E2EAITrainer(model, tiny_config)
    trainer.train_batch(batch)
    expected = _strict_output(model, batch)
    checkpoint = trainer.save_checkpoint(Path(tiny_config.logging.output_dir) / "roundtrip.pt", epoch=0)

    restored_model = E2EAIModel(tiny_config.model)
    restored_trainer = E2EAITrainer(restored_model, tiny_config)
    restored_trainer.load_checkpoint(checkpoint, load_optimizer=False, restore_rng=False)
    actual = _strict_output(restored_model, batch)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert torch.equal(
        restored_model.directional_buffer.deep_factor_ic_sum,
        model.directional_buffer.deep_factor_ic_sum,
    )
    trainer.close()
    restored_trainer.close()

