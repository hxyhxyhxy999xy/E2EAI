"""Ensure source-tree imports work before an editable installation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tiny_config(tmp_path):
    from e2eai.config import ExperimentConfig

    config = ExperimentConfig()
    config.seed = 13
    config.device = "cpu"
    config.model.num_factors = 4
    config.synthetic.num_factors = 4
    config.synthetic.num_dates = 8
    config.synthetic.num_stocks = 8
    config.synthetic.num_industries = 2
    config.synthetic.active_probability = 1.0
    config.model.context.hidden_dim = 4
    config.model.context.dropout = 0.0
    config.model.gat.dropout = 0.0
    config.model.factor_selection.hidden_dim = 6
    config.model.factor_selection.gamma_f = 0.10
    config.model.portfolio.gamma_p = 0.20
    config.model.portfolio.theta = 0.25
    config.model.portfolio.min_selected_stocks = 2
    config.local_optimizer.lr = 0.1
    config.local_optimizer.n_iter = 1
    config.training.batch_dates = 3
    config.training.epochs = 1
    config.training.early_stopping_patience = 2
    config.training.validation_target = "validation_loss"
    config.data.train_ratio = 0.50
    config.data.validation_ratio = 0.25
    config.data.test_ratio = 0.25
    config.logging.tensorboard = False
    config.logging.wandb = False
    config.logging.output_dir = str(tmp_path / "outputs")
    config.validate()
    return config


@pytest.fixture
def tiny_loaders(tiny_config):
    from e2eai.data import generate_synthetic_frame
    from e2eai.data.loaders import build_dataloaders

    synthetic = tiny_config.synthetic
    frame = generate_synthetic_frame(
        synthetic.num_dates,
        synthetic.num_stocks,
        synthetic.num_factors,
        synthetic.num_industries,
        tiny_config.model.horizons,
        tiny_config.seed,
        synthetic.active_probability,
        synthetic.signal_noise,
    )
    return build_dataloaders(tiny_config, frame)

