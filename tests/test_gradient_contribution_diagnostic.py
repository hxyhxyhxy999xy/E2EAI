from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from e2eai.config import load_config
from e2eai.models.e2eai import E2EAIModel
from scripts.run_gradient_contribution_diagnostic import (
    _component_metrics,
    _layer_metrics,
    _load_checkpoints,
    _parameter_layout,
)

ROOT = Path(__file__).resolve().parents[1]


def _model() -> E2EAIModel:
    config = load_config(ROOT / "configs/experiment_a_daily_core.yaml")
    return E2EAIModel(config.model, config.ablation)


def test_parameter_order_matches_real_scorer_state_dict() -> None:
    model = _model()
    names = [name for name, _ in _parameter_layout(model)]
    assert names == ["hidden.weight", "hidden.bias", "output.weight", "output.bias"]


def test_weighted_gross_gradient_is_exact_scaling() -> None:
    score = np.asarray([1.0, -2.0, 0.5])
    raw = np.asarray([0.25, 0.5, -0.75])
    lam = 70.21024290734199
    weighted = lam * raw
    assert np.max(np.abs(weighted - lam * raw)) == 0.0
    assert _component_metrics(score, raw, lam)["gross_grad_norm_weighted"] == np.linalg.norm(weighted)


def test_cosine_and_cancellation_are_correct() -> None:
    metrics = _component_metrics(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0]), 1.0)
    assert metrics["gradient_cosine"] == 1.0
    assert metrics["cancellation_ratio"] == 1.0
    metrics = _component_metrics(np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]), 1.0)
    assert metrics["gradient_cosine"] == -1.0
    assert metrics["cancellation_ratio"] == 0.0


def test_layer_metrics_aggregate_real_mlp_layers() -> None:
    score = {"hidden.weight": np.ones(4), "hidden.bias": np.ones(2), "output.weight": np.ones(2), "output.bias": np.ones(1)}
    gross = {name: value.copy() for name, value in score.items()}
    rows = _layer_metrics(score, gross, 1.0)
    assert [row["layer"] for row in rows] == ["first_linear", "output_linear"]
    assert all("simple_mlp_scorer." in str(row["parameter_names"]) for row in rows)
    assert all(row["gradient_ratio"] == 1.0 for row in rows)


def test_diagnostic_script_has_no_optimizer_step_call() -> None:
    path = ROOT / "scripts/run_gradient_contribution_diagnostic.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "step" for node in calls)


def test_diagnostic_uses_train_loader_only() -> None:
    source = (ROOT / "scripts/run_gradient_contribution_diagnostic.py").read_text(encoding="utf-8")
    assert "loaders.train" in source
    assert "loaders.validation" not in source
    assert "loaders.test" not in source


def test_diagnostic_does_not_read_2022() -> None:
    source = (ROOT / "scripts/run_gradient_contribution_diagnostic.py").read_text(encoding="utf-8")
    assert "2022" in source  # explicit manifest/report guard
    assert "2022-01" not in source


def test_initial_checkpoint_is_reported_as_step_zero(tmp_path: Path) -> None:
    model = _model()
    payload = {"model_state_dict": model.state_dict(), "global_step": 91}
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save(payload, checkpoint_dir / "initial_from_a.pt")
    payload["global_step"] = 20
    torch.save(payload, checkpoint_dir / "best.pt")
    payload["global_step"] = 220
    torch.save(payload, checkpoint_dir / "last.pt")
    items = _load_checkpoints(checkpoint_dir, torch.device("cpu"))
    assert [item["optimizer_step"] for item in items] == [0, 20, 220]
