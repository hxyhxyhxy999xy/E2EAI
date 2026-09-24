from __future__ import annotations

import copy

import pytest
import torch

from e2eai.config import AblationConfig, ExperimentConfig, load_config
from e2eai.evaluation.daily_validation import evaluate_daily_weight_path
from e2eai.models.e2eai import E2EAIModel
from e2eai.models.portfolio import GatedPortfolioAllocator
from e2eai.data.dataset import generate_synthetic_frame
from e2eai.data.loaders import build_dataloaders
from e2eai.training.trainer import E2EAITrainer
from scripts.run_daily_experiment import (
    _model_state_checksum,
    assert_evaluation_only_compatible,
)


def test_experiment_a_inherits_daily_strategy_config() -> None:
    config = load_config("configs/experiments/experiment_a.yaml")
    assert config.experiment_line == "daily_strategy"
    assert config.model.horizons == [2, 3, 5, 10, 15, 20]
    assert config.model.execution_horizon == 2
    assert config.training.optimization_horizons == [2]
    assert config.training.validation_target == "validation_daily_sharpe"
    assert not config.ablation.use_factor_selector
    assert config.loss.lambda_portfolio == 0.0
    assert config.loss.lambda_score == 1.0


def test_daily_experiment_definitions_are_explicit() -> None:
    experiment_b = load_config("configs/experiments/experiment_b.yaml")
    experiment_c1 = load_config("configs/experiments/experiment_c1.yaml")
    experiment_c2 = load_config("configs/experiments/experiment_c2.yaml")
    experiment_d = load_config("configs/experiments/experiment_d.yaml")
    assert experiment_b.experiment.mode == "evaluation_only"
    assert experiment_b.experiment.source_experiment == "experiment_a"
    assert experiment_c1.model.factor_selection.gate_mode_train == "soft"
    assert experiment_c1.model.factor_selection.gate_mode_eval == "soft"
    assert experiment_c2.model.factor_selection.gate_mode_train == "ste_hard"
    assert experiment_c2.model.factor_selection.gate_mode_eval == "hard"
    assert experiment_d.ablation.use_industry_gat
    assert not experiment_d.ablation.use_universe_gat
    assert experiment_d.experiment.selection_candidates == ["experiment_c1", "experiment_c2"]


def test_topk_equal_weight_uses_dynamic_valid_count() -> None:
    allocator = GatedPortfolioAllocator(
        hidden_dim=4,
        num_horizons=1,
        allocation_mode="topk_equal_weight",
        topk_ratio=0.5,
    )
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 0.0, 0.0]])
    output = allocator(
        torch.zeros(2, 4, 4),
        torch.zeros(2, 1, 4),
        torch.ones(1),
        mask,
        portfolio_scores=scores,
    )
    assert output.stock_mask.sum(dim=-1).tolist() == [[2], [1]]
    assert torch.allclose(output.portfolio_weights.sum(dim=-1), torch.ones(2, 1))
    assert torch.allclose(output.portfolio_weights[0, 0, :2], torch.tensor([0.5, 0.5]))


def test_ablation_forward_preserves_shapes() -> None:
    config = ExperimentConfig()
    config.model.num_factors = 4
    config.model.horizons = [2, 3]
    config.model.execution_horizon = 2
    config.model.strategy_head = "simple_mlp"
    config.model.context.hidden_dim = 4
    config.model.portfolio.allocation_mode = "topk_equal_weight"
    config.model.portfolio.topk_ratio = 0.5
    config.synthetic.num_factors = 4
    ablation = AblationConfig(
        use_factor_selector=False,
        use_industry_gat=False,
        use_universe_gat=False,
        use_direction_buffer=False,
        use_directional_approximation=False,
    )
    model = E2EAIModel(config.model, ablation)
    factors = torch.randn(2, 6, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    graph = torch.eye(6, dtype=torch.bool).expand(2, -1, -1)
    output = model(factors, mask, graph, graph)
    assert output.portfolio_weights.shape == (2, 2, 6)
    assert output.deep_factors.shape == (2, 2, 6)
    assert torch.allclose(output.selected_factors, factors)
    assert torch.count_nonzero(output.deep_factor_approx) == 0


def test_daily_metrics_use_drifted_previous_weights_and_cost() -> None:
    targets = [
        {"A": 0.5, "B": 0.5},
        {"A": 0.5, "B": 0.5},
    ]
    returns = [
        {"A": 0.10, "B": 0.0},
        {"A": 0.0, "B": 0.0},
    ]
    gross = evaluate_daily_weight_path(targets, returns, cost_bps=0.0)
    net = evaluate_daily_weight_path(targets, returns, cost_bps=10.0)
    assert gross["validation_daily_turnover"] > 0.0
    assert net["validation_daily_return"] < gross["validation_daily_return"]
    assert gross["validation_effective_n"] == 2.0


def test_experiment_a_has_a_learnable_daily_training_path(tmp_path) -> None:
    config = ExperimentConfig()
    config.experiment_line = "daily_strategy"
    config.device = "cpu"
    config.model.num_factors = 4
    config.model.horizons = [2, 3]
    config.model.execution_horizon = 2
    config.model.strategy_head = "simple_mlp"
    config.model.context.hidden_dim = 4
    config.model.context.dropout = 0.0
    config.model.gat.dropout = 0.0
    config.model.portfolio.allocation_mode = "topk_equal_weight"
    config.model.portfolio.topk_ratio = 0.5
    config.ablation = AblationConfig(False, False, False, False, False)
    config.loss.lambda_s = 0.0
    config.loss.lambda_f = 0.0
    config.loss.lambda_e = 0.0
    config.loss.lambda_portfolio = 0.0
    config.loss.lambda_score = 1.0
    config.training.optimization_horizons = [2]
    config.training.validation_target = "validation_daily_sharpe"
    config.training.batch_dates = 2
    config.training.epochs = 1
    config.data.train_ratio = 0.5
    config.data.validation_ratio = 0.25
    config.data.test_ratio = 0.25
    config.synthetic.num_factors = 4
    config.synthetic.num_dates = 8
    config.synthetic.num_stocks = 8
    config.synthetic.num_industries = 2
    config.synthetic.active_probability = 1.0
    config.logging.tensorboard = False
    config.logging.output_dir = str(tmp_path / "daily")
    config.validate()
    frame = generate_synthetic_frame(
        num_dates=8,
        num_stocks=8,
        num_factors=4,
        num_industries=2,
        horizons=config.model.horizons,
        active_probability=1.0,
    )
    frame[config.data.execution_return_column] = frame["forward_return_2"]
    loaders = build_dataloaders(config, frame)
    model = E2EAIModel(config.model, config.ablation)
    trainer = E2EAITrainer(model, config)
    try:
        fit = trainer.fit(loaders.train, loaders.validation, epochs=1)
    finally:
        trainer.close()
    assert fit.best_epoch == 0
    assert "validation_daily_sharpe" in fit.history[0]
    assert "train_score_prediction_loss" in fit.history[0]
    assert "train_weighted_portfolio_loss" in fit.history[0]
    assert fit.history[0]["train_weighted_portfolio_loss"] == 0.0
    assert not trainer.last_output.portfolio_weights.requires_grad
    assert any(
        parameter.grad is not None and (parameter.grad != 0).any()
        for parameter in model.simple_mlp_scorer.parameters()
    )


def test_experiment_b_reuses_a_checkpoint_scores_and_parameters(tmp_path) -> None:
    source = ExperimentConfig()
    source.experiment_line = "daily_strategy"
    source.device = "cpu"
    source.model.num_factors = 4
    source.synthetic.num_factors = 4
    source.model.horizons = [2]
    source.model.execution_horizon = 2
    source.model.strategy_head = "simple_mlp"
    source.model.context.hidden_dim = 4
    source.model.context.dropout = 0.0
    source.model.gat.dropout = 0.0
    source.model.portfolio.allocation_mode = "topk_equal_weight"
    source.model.portfolio.topk_ratio = 0.25
    source.ablation = AblationConfig(False, False, False, False, False)
    source.loss.lambda_portfolio = 0.0
    source.loss.lambda_score = 1.0
    source.loss.lambda_s = source.loss.lambda_f = source.loss.lambda_e = 0.0
    source.logging.tensorboard = False

    evaluation = copy.deepcopy(source)
    evaluation.model.portfolio.allocation_mode = "capped_simplex"
    evaluation.model.portfolio.theta = 0.25
    evaluation.model.portfolio.gamma_p = 0.0
    evaluation.experiment.mode = "evaluation_only"
    evaluation.experiment.source_experiment = "experiment_a"
    assert_evaluation_only_compatible(source, evaluation)

    torch.manual_seed(11)
    source_model = E2EAIModel(source.model, source.ablation).eval()
    evaluation_model = E2EAIModel(evaluation.model, evaluation.ablation).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state_dict": source_model.state_dict(), "best_epoch": 3}, checkpoint)
    source_trainer = E2EAITrainer(source_model, source, create_optimizer=False)
    evaluation_trainer = E2EAITrainer(
        evaluation_model, evaluation, create_optimizer=False
    )
    try:
        evaluation_trainer.load_checkpoint(
            checkpoint, load_optimizer=False, restore_rng=False
        )
        assert evaluation_trainer.optimizer is None
        with pytest.raises(RuntimeError, match="evaluation only"):
            evaluation_trainer.train_batch({})
        assert _model_state_checksum(source_model) == _model_state_checksum(evaluation_model)

        factors = torch.randn(2, 8, 4)
        mask = torch.ones(2, 8, dtype=torch.bool)
        graph = torch.eye(8, dtype=torch.bool).expand(2, -1, -1)
        batch = {
            "raw_factors": factors,
            "asset_mask": mask,
            "industry_adjacency": graph,
            "universe_adjacency": graph,
            "dates": ["2021-01-04", "2021-01-05"],
            "asset_ids": [[f"S{i}" for i in range(8)] for _ in range(2)],
        }
        source_records, source_hash = source_trainer.collect_raw_scores([batch])
        evaluation_records, evaluation_hash = evaluation_trainer.collect_raw_scores([batch])
        assert source_records == evaluation_records
        assert source_hash == evaluation_hash

        with torch.no_grad():
            source_output = source_model(factors, mask, graph, graph)
            evaluation_output = evaluation_model(factors, mask, graph, graph)
        for name, source_value in source_output.tensor_dict().items():
            if name in {"stock_selection_mask", "portfolio_weights"}:
                continue
            assert torch.allclose(source_value, evaluation_output.tensor_dict()[name])
        assert not torch.equal(
            source_output.stock_selection_mask,
            evaluation_output.stock_selection_mask,
        )
        assert not torch.allclose(
            source_output.portfolio_weights,
            evaluation_output.portfolio_weights,
        )
    finally:
        source_trainer.close()
        evaluation_trainer.close()


def test_gat_diagnostics_separate_neutral_and_component_norms() -> None:
    config = ExperimentConfig()
    config.device = "cpu"
    config.model.num_factors = 4
    config.synthetic.num_factors = 4
    config.model.horizons = [2]
    config.model.execution_horizon = 2
    config.data.panel_return_horizons = [2]
    config.data.label_exit_offsets = [2]
    config.model.strategy_head = "simple_mlp"
    config.model.context.hidden_dim = 4
    config.model.context.dropout = 0.0
    config.model.gat.dropout = 0.0
    config.model.portfolio.allocation_mode = "topk_equal_weight"
    config.logging.tensorboard = False
    ablation = AblationConfig(False, True, False, False, False)
    model = E2EAIModel(config.model, ablation).eval()
    trainer = E2EAITrainer(model, config, create_optimizer=False)
    try:
        factors = torch.randn(2, 6, 4)
        mask = torch.ones(2, 6, dtype=torch.bool)
        graph = torch.ones(2, 6, 6, dtype=torch.bool)
        with torch.no_grad():
            output = model(factors, mask, graph, graph)
        diagnostics = trainer._representation_diagnostics(output, {"asset_mask": mask})
        context_norm = torch.linalg.vector_norm(output.stock_context)
        industry_norm = torch.linalg.vector_norm(output.industry_neutral)
        component_norm = torch.linalg.vector_norm(output.industry_gat_component)
        assert diagnostics["industry_neutral_norm_ratio"] == pytest.approx(
            float(industry_norm / context_norm)
        )
        assert diagnostics["industry_gat_component_norm_ratio"] == pytest.approx(
            float(component_norm / context_norm)
        )
        assert "industry_gat_norm_ratio" not in diagnostics
    finally:
        trainer.close()


@pytest.mark.parametrize(
    ("experiment_name", "expected", "inactive"),
    [
        (
            "experiment_a",
            {"simple_mlp_scorer"},
            {"factor_selector", "neutralization.industry_block", "portfolio_allocator"},
        ),
        (
            "experiment_c1",
            {"factor_selector", "simple_mlp_scorer"},
            {"neutralization.industry_block", "portfolio_allocator"},
        ),
        (
            "experiment_c2",
            {"factor_selector", "simple_mlp_scorer"},
            {"neutralization.industry_block", "portfolio_allocator"},
        ),
        (
            "experiment_d",
            {"factor_selector", "simple_mlp_scorer", "neutralization.industry_block"},
            {"neutralization.universe_block", "portfolio_allocator"},
        ),
    ],
)
def test_daily_experiment_gradient_routes(
    tmp_path,
    experiment_name: str,
    expected: set[str],
    inactive: set[str],
) -> None:
    config = load_config(f"configs/experiments/{experiment_name}.yaml")
    config.device = "cpu"
    config.model.num_factors = 4
    config.synthetic.num_factors = 4
    config.model.horizons = [2]
    config.model.execution_horizon = 2
    config.data.panel_return_horizons = [2]
    config.data.label_exit_offsets = [2]
    config.model.context.hidden_dim = 4
    config.model.context.dropout = 0.0
    config.model.gat.dropout = 0.0
    config.model.factor_selection.hidden_dim = 4
    config.model.portfolio.allocation_mode = "topk_equal_weight"
    config.model.portfolio.topk_ratio = 0.25
    config.local_optimizer.n_iter = 1
    config.logging.tensorboard = False
    config.logging.output_dir = str(tmp_path / experiment_name)
    config.experiment.selection_candidates = []
    config.validate()

    torch.manual_seed(101)
    batch_size, stocks = 3, 8
    factors = torch.randn(batch_size, stocks, 4)
    returns = factors[..., 0].unsqueeze(1) + 0.1 * torch.randn(batch_size, 1, stocks)
    mask = torch.ones(batch_size, stocks, dtype=torch.bool)
    graph = torch.ones(batch_size, stocks, stocks, dtype=torch.bool)
    batch = {
        "raw_factors": factors,
        "forward_returns": returns,
        "return_mask": mask.unsqueeze(1),
        "asset_mask": mask,
        "industry_adjacency": graph,
        "universe_adjacency": graph,
    }
    model = E2EAIModel(config.model, config.ablation)
    trainer = E2EAITrainer(model, config)
    try:
        diagnostics = trainer.train_batch(batch)
        assert diagnostics["weighted_portfolio_loss"] == 0.0
        gradient_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and (parameter.grad != 0).any()
        }
        for prefix in expected:
            assert any(name.startswith(prefix) for name in gradient_names), prefix
        for prefix in inactive:
            assert not any(name.startswith(prefix) for name in gradient_names), prefix
    finally:
        trainer.close()
