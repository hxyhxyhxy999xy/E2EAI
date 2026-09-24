from pathlib import Path

import pytest

from e2eai.config import ExperimentConfig, load_config
from e2eai.workflows import PAPER_REPRO_DISABLED_MESSAGE, load_configured_data


ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_load_and_validate() -> None:
    default = load_config(ROOT / "configs" / "default.yaml")
    demo = load_config(ROOT / "configs" / "synthetic_demo.yaml")
    assert default.model.horizons == [3, 5, 10, 15, 20]
    assert demo.model.num_factors == demo.synthetic.num_factors == 8


def test_invalid_cap_is_rejected() -> None:
    config = ExperimentConfig()
    config.model.portfolio.theta = 0.0
    with pytest.raises(ValueError, match="theta"):
        config.validate()


def test_legacy_gate_mode_maps_to_train_and_hard_eval(tmp_path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "model:\n  factor_selection:\n    gate_mode: ste_hard\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model.factor_selection.gate_mode_train == "ste_hard"
    assert config.model.factor_selection.gate_mode_eval == "hard"


def test_unimplemented_daily_active_ir_cannot_be_validation_target() -> None:
    config = ExperimentConfig()
    config.experiment_line = "daily_strategy"
    config.model.execution_horizon = config.model.horizons[0]
    config.training.validation_target = "validation_daily_active_ir"
    with pytest.raises(
        ValueError,
        match="validation_daily_active_ir is not implemented for daily_strategy",
    ):
        config.validate()


def test_removed_legacy_paper_panel_has_an_explicit_disabled_error() -> None:
    config = load_config(ROOT / "configs" / "paper_repro.yaml")
    with pytest.raises(RuntimeError, match="paper_repro disabled") as error:
        load_configured_data(config)
    assert str(error.value) == PAPER_REPRO_DISABLED_MESSAGE
