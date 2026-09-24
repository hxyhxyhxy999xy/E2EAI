"""Typed experiment configuration and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


@dataclass
class DataConfig:
    path: str | None = None
    format: Literal["auto", "long", "alpha_panel"] = "auto"
    date_column: str = "date"
    asset_column: str = "asset_id"
    industry_column: str = "industry_id"
    factor_columns: list[str] = field(default_factory=list)
    factor_prefix: str = "factor_"
    return_prefix: str = "forward_return_"
    benchmark_prefix: str = "benchmark_return_"
    missing_factor_strategy: Literal["cross_sectional_median", "zero_after_standardization"] = (
        "cross_sectional_median"
    )
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    train_start: str | None = None
    train_end: str | None = None
    validation_start: str | None = None
    validation_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    universe_graph_mode: Literal["all_valid", "cross_industry_only"] = "all_valid"
    unknown_industry_self_only: bool = True
    factor_groups: dict[str, list[str]] = field(default_factory=dict)
    alpha_key: str = "alpha_tensor"
    return_key: str = "forward_returns"
    benchmark_key: str = "benchmark_returns"
    return_path: str | None = None
    execution_return_path: str | None = None
    execution_return_key: str = "execution_1d_return"
    execution_return_column: str = "execution_1d_return"
    benchmark_path: str | None = None
    panel_return_horizons: list[int] = field(default_factory=list)
    label_definition: Literal["paper_formula", "legacy_t_plus_1"] = "paper_formula"
    label_exit_offsets: list[int] = field(default_factory=list)
    purge_label_overlap: bool = False
    # Deliberately permit a train+validation-only panel for daily experiments.
    # The split builder purges trailing validation labels rather than creating
    # or evaluating a test split.
    daily_validation_only: bool = False
    eligibility_key: str = "eligible_mask"
    factor_validity_key: str = "factor_valid_mask"
    alpha_names_key: str = "alpha_names"
    alpha_directions_key: str = "alpha_directions"
    industry_ids_key: str = "industry_ids"
    market_cap_key: str = "free_float_market_cap"
    industry_npz_path: str | None = None
    index_universe: str | None = None
    membership_path: str | None = None
    factor_selection_manifest_path: str | None = None
    factor_directions_path: str | None = None
    apply_factor_direction: bool = False
    mmap_mode: Literal["r", "c"] | None = "r"
    max_assets_per_date: int | None = None
    asset_selection: Literal["axis_order", "market_cap"] = "axis_order"
    single_return_policy: Literal["require_single_horizon", "broadcast"] = (
        "require_single_horizon"
    )


@dataclass
class FactorSelectionConfig:
    hidden_dim: int = 32
    gamma_f: float = 0.02
    pooling: Literal["mean", "mean_std"] = "mean"
    gate_mode_train: Literal["ste_hard", "soft", "hard"] = "ste_hard"
    gate_mode_eval: Literal["ste_hard", "soft", "hard"] = "hard"
    min_selected_factors: int = 1

    @property
    def gate_mode(self) -> Literal["ste_hard", "soft", "hard"]:
        """Backward-compatible programmatic alias for the training gate."""
        return self.gate_mode_train

    @gate_mode.setter
    def gate_mode(self, value: Literal["ste_hard", "soft", "hard"]) -> None:
        self.gate_mode_train = value


@dataclass
class ContextEncoderConfig:
    hidden_dim: int = 64
    normalization: Literal["cross_sectional", "torch_batchnorm"] = "cross_sectional"
    dropout: float = 0.1
    eps: float = 1e-6


@dataclass
class GATConfig:
    heads: int = 1
    dropout: float = 0.1
    leaky_relu_slope: float = 0.2
    add_self_loops: bool = True


@dataclass
class DirectionConfig:
    normalization: Literal["none", "cross_factor_zscore"] = "cross_factor_zscore"
    min_stocks: int = 3
    eps: float = 1e-6
    zero_direction: Literal[-1, 1] = 1
    zscore_approximation_input: bool = False
    reset_each_epoch: bool = False


@dataclass
class PortfolioConfig:
    allocation_mode: Literal[
        "legacy_cap", "capped_simplex", "topk_equal_weight", "long_only_softmax"
    ] = "legacy_cap"
    gamma_p: float = 0.01
    gamma_p_mode: Literal["fixed", "inverse_n"] = "fixed"
    # Deprecated for long_only_softmax; retained only for legacy cap experiments.
    theta: float = 0.10
    min_selected_stocks: int = 1
    cap_mode_train: Literal["penalty_only", "capped_simplex"] = "penalty_only"
    cap_mode_eval: Literal["penalty_only", "capped_simplex"] = "capped_simplex"
    projection_iterations: int = 64
    topk_ratio: float = 0.20
    topk: int | None = None


@dataclass
class AblationConfig:
    """Structural switches shared by paper and daily-strategy experiments."""

    use_factor_selector: bool = True
    use_industry_gat: bool = True
    use_universe_gat: bool = True
    use_direction_buffer: bool = True
    use_directional_approximation: bool = True


@dataclass
class ModelConfig:
    num_factors: int = 16
    horizons: list[int] = field(default_factory=lambda: [3, 5, 10, 15, 20])
    execution_horizon: int | None = None
    strategy_head: Literal["e2eai", "simple_mlp", "factor_mean"] = "e2eai"
    factor_selection: FactorSelectionConfig = field(default_factory=FactorSelectionConfig)
    context: ContextEncoderConfig = field(default_factory=ContextEncoderConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    share_deep_factor_head: bool = False


@dataclass
class LossConfig:
    lambda_portfolio: float = 1.0
    lambda_s: float = 0.1
    lambda_f: float = 0.1
    lambda_e: float = 0.1
    lambda_up: float = 1.0
    lambda_score: float = 0.0
    eps: float = 1e-6


@dataclass
class LocalOptimizerConfig:
    lr: float = 0.05
    n_iter: int = 10
    differentiable: bool = True


@dataclass
class GlobalOptimizerConfig:
    name: Literal["AdamW", "Adam"] = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: Literal["none", "cosine"] = "none"


@dataclass
class TrainingConfig:
    batch_dates: int = 16
    epochs: int = 50
    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 10
    validation_target: Literal[
        "validation_ir",
        "validation_excess_return",
        "validation_loss",
        "validation_daily_sharpe",
        "validation_daily_sharpe_7_5bps",
        "validation_daily_return",
        "validation_daily_active_ir",
    ] = "validation_ir"
    optimization_horizons: list[int] = field(default_factory=list)
    mixed_precision: bool = False
    shuffle_dates: bool = False
    num_workers: int = 0


@dataclass
class EvaluationConfig:
    mode: Literal["fixed_model", "walk_forward"] = "fixed_model"
    backtest_mode: Literal["overlapping_label", "non_overlapping"] = "overlapping_label"
    annual_trading_days: int = 252
    risk_free_rate: float = 0.0
    holding_threshold: float = 1e-8
    cost_bps: float = 0.0
    cost_scenarios_bps: list[float] = field(default_factory=lambda: [0.0, 5.0, 10.0])


@dataclass
class LoggingConfig:
    output_dir: str = "output/default"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "e2eai"


@dataclass
class SyntheticConfig:
    num_dates: int = 40
    num_stocks: int = 40
    num_factors: int = 16
    num_industries: int = 8
    active_probability: float = 0.90
    signal_noise: float = 0.35


@dataclass
class ExperimentRunConfig:
    """Execution semantics that do not change the model architecture."""

    mode: Literal["train", "evaluation_only"] = "train"
    source_experiment: str | None = None
    selection_candidates: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    experiment_line: Literal["paper_repro", "daily_strategy"] = "paper_repro"
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    local_optimizer: LocalOptimizerConfig = field(default_factory=LocalOptimizerConfig)
    global_optimizer: GlobalOptimizerConfig = field(default_factory=GlobalOptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    experiment: ExperimentRunConfig = field(default_factory=ExperimentRunConfig)

    def validate(self) -> None:
        """Raise ``ValueError`` for internally inconsistent settings."""
        if self.model.num_factors < 1:
            raise ValueError("model.num_factors must be positive")
        if not self.model.horizons or any(h <= 0 for h in self.model.horizons):
            raise ValueError("model.horizons must contain positive integers")
        if len(set(self.model.horizons)) != len(self.model.horizons):
            raise ValueError("model.horizons must not contain duplicates")
        if self.model.execution_horizon is not None:
            if self.model.execution_horizon < 1:
                raise ValueError("model.execution_horizon must be positive")
            if self.model.execution_horizon not in self.model.horizons:
                raise ValueError("model.execution_horizon must be present in model.horizons")
        fs = self.model.factor_selection
        if not 0.0 <= fs.gamma_f <= 1.0:
            raise ValueError("gamma_f must be in [0, 1]")
        valid_gate_modes = {"soft", "ste_hard", "hard"}
        if fs.gate_mode_train not in valid_gate_modes or fs.gate_mode_eval not in valid_gate_modes:
            raise ValueError("factor gate modes must be soft, ste_hard, or hard")
        if not 1 <= fs.min_selected_factors <= self.model.num_factors:
            raise ValueError("min_selected_factors must be in [1, num_factors]")
        portfolio = self.model.portfolio
        if portfolio.gamma_p_mode not in {"fixed", "inverse_n"}:
            raise ValueError("gamma_p_mode must be 'fixed' or 'inverse_n'")
        if portfolio.gamma_p < 0.0:
            raise ValueError("gamma_p must be non-negative")
        if portfolio.gamma_p_mode == "fixed" and portfolio.gamma_p > 1.0:
            raise ValueError("fixed gamma_p must be in [0, 1]")
        if portfolio.allocation_mode not in {
            "legacy_cap", "capped_simplex", "topk_equal_weight", "long_only_softmax"
        }:
            raise ValueError('Unknown portfolio allocation_mode')
        if portfolio.allocation_mode in {"legacy_cap", "capped_simplex"} and not 0.0 < portfolio.theta <= 1.0:
            raise ValueError("theta must be in (0, 1]")
        if portfolio.allocation_mode == 'long_only_softmax' and portfolio.min_selected_stocks != 1:
            raise ValueError('No-cap allocation only allows the one-stock safety fallback')
        if not 0.0 < portfolio.topk_ratio <= 1.0:
            raise ValueError("portfolio.topk_ratio must be in (0, 1]")
        if portfolio.topk is not None and portfolio.topk < 1:
            raise ValueError("portfolio.topk must be positive when set")
        if self.data.index_universe is not None:
            if self.data.index_universe not in ('csi300', 'csi500', 'csi1000'):
                raise ValueError('Unknown index_universe')
            if not self.data.membership_path:
                raise ValueError('Historical membership_path is required; no fallback allowed')
            if self.data.max_assets_per_date is not None or self.data.asset_selection != 'axis_order':
                raise ValueError('Index universes cannot be truncated or ranked by market cap')
        if portfolio.min_selected_stocks < 1:
            raise ValueError("min_selected_stocks must be positive")
        if self.local_optimizer.n_iter < 1 or self.local_optimizer.lr <= 0:
            raise ValueError("local optimizer requires n_iter >= 1 and lr > 0")
        if self.training.batch_dates < 1 or self.training.epochs < 1:
            raise ValueError("batch_dates and epochs must be positive")
        unknown_optimization = set(self.training.optimization_horizons) - set(self.model.horizons)
        if unknown_optimization:
            raise ValueError(
                "training.optimization_horizons must be contained in model.horizons; "
                f"unknown={sorted(unknown_optimization)}"
            )
        if self.training.validation_target.startswith("validation_daily_"):
            if self.model.execution_horizon is None:
                raise ValueError("daily validation targets require model.execution_horizon")
        if (
            self.experiment_line == "daily_strategy"
            and self.training.validation_target == "validation_daily_active_ir"
        ):
            raise ValueError(
                "validation_daily_active_ir is not implemented for daily_strategy"
            )
        if self.evaluation.cost_bps < 0 or any(value < 0 for value in self.evaluation.cost_scenarios_bps):
            raise ValueError("transaction costs must be non-negative")
        loss_weights = (
            self.loss.lambda_portfolio,
            self.loss.lambda_score,
            self.loss.lambda_s,
            self.loss.lambda_f,
            self.loss.lambda_e,
            self.loss.lambda_up,
        )
        if any(value < 0 for value in loss_weights):
            raise ValueError("loss weights must be non-negative")
        if self.experiment.mode == "evaluation_only" and not self.experiment.source_experiment:
            raise ValueError("evaluation_only experiments require source_experiment")
        if self.experiment.mode == "train" and self.experiment.source_experiment is not None:
            raise ValueError("source_experiment is only valid for evaluation_only experiments")
        if self.experiment.mode == "evaluation_only" and self.experiment.selection_candidates:
            raise ValueError("evaluation_only experiments cannot select a training candidate")
        if self.data.max_assets_per_date is not None and self.data.max_assets_per_date < 1:
            raise ValueError("data.max_assets_per_date must be positive when set")
        if any(value <= 0 for value in self.data.panel_return_horizons):
            raise ValueError("data.panel_return_horizons must contain positive integers")
        if self.data.label_definition not in {"paper_formula", "legacy_t_plus_1"}:
            raise ValueError("Unsupported data.label_definition")
        if self.data.label_exit_offsets:
            if len(self.data.label_exit_offsets) != len(self.model.horizons):
                raise ValueError(
                    "data.label_exit_offsets must have one entry per model horizon"
                )
            if any(value < 1 for value in self.data.label_exit_offsets):
                raise ValueError("data.label_exit_offsets must be positive")
        ratios = (
            self.data.train_ratio,
            self.data.validation_ratio,
            self.data.test_ratio,
        )
        if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("chronological split ratios must be nonnegative and sum to one")
        if self.data.path is None and self.synthetic.num_factors != self.model.num_factors:
            raise ValueError("synthetic.num_factors must equal model.num_factors")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly dictionary."""
        return asdict(self)


def _construct_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    valid_names = {item.name for item in fields(cls)}
    unknown = set(values) - valid_names
    if unknown:
        raise ValueError(f"Unknown configuration keys for {cls.__name__}: {sorted(unknown)}")
    for item in fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        hinted = hints.get(item.name)
        if isinstance(hinted, type) and is_dataclass(hinted) and isinstance(value, dict):
            value = _construct_dataclass(hinted, value)
        kwargs[item.name] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge YAML mappings without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_base(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    chain = set() if seen is None else set(seen)
    if resolved in chain:
        raise ValueError(f"Circular base_config chain at {resolved}")
    chain.add(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping")
    base_name = raw.pop("base_config", None)
    if base_name is None:
        return raw
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return _deep_merge(_load_yaml_with_base(base_path, chain), raw)


def _migrate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Map legacy single factor-gate YAML settings onto explicit train/eval modes."""
    migrated = dict(raw)
    model = migrated.get("model")
    if not isinstance(model, dict):
        return migrated
    model = dict(model)
    factor_selection = model.get("factor_selection")
    if not isinstance(factor_selection, dict) or "gate_mode" not in factor_selection:
        migrated["model"] = model
        return migrated
    factor_selection = dict(factor_selection)
    legacy_mode = factor_selection.pop("gate_mode")
    if (
        "gate_mode_train" in factor_selection
        and factor_selection["gate_mode_train"] != legacy_mode
    ):
        raise ValueError("legacy gate_mode conflicts with gate_mode_train")
    factor_selection.setdefault("gate_mode_train", legacy_mode)
    # Historical evaluation behavior was always an exact hard gate.
    factor_selection.setdefault("gate_mode_eval", "hard")
    model["factor_selection"] = factor_selection
    migrated["model"] = model
    return migrated


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""
    config_path = Path(path)
    raw = _migrate_legacy_config(_load_yaml_with_base(config_path))
    config = _construct_dataclass(ExperimentConfig, raw)
    config.validate()
    return config
