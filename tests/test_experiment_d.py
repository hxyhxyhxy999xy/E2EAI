from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from e2eai.evaluation.daily_validation import _simulate_daily_weight_path
from e2eai.training.net_return import account_sequence_torch, numpy_drifted_rows
from scripts.run_daily_experiment_d import LAMBDA_PORTFOLIO, _batch_boundary_test

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    assets = [["a", "b", "c"]] * 4
    weights = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.35, 0.45, 0.2], [0.3, 0.5, 0.2]], dtype=torch.float32, requires_grad=True)
    returns = torch.tensor([[0.01, -0.02, 0.03], [0.02, 0.01, -0.01], [-0.01, 0.03, 0.02], [0.04, -0.01, 0.01]], dtype=torch.float32)
    mask = torch.ones_like(weights, dtype=torch.bool)
    return assets, weights, returns, mask


def test_toy_accounting_matches_independent_numpy_and_formal_path() -> None:
    assets, weights, returns, mask = _toy()
    targets = [{asset: float(value) for asset, value in zip(assets[0], row.detach().numpy())} for row in weights]
    returns_maps = [{asset: float(value) for asset, value in zip(assets[0], row.numpy())} for row in returns]
    torch_result = account_sequence_torch(weights, assets, mask, returns, mask)
    drift, gross, turnover, cost, net = numpy_drifted_rows(targets, returns_maps)
    formal = _simulate_daily_weight_path(targets, returns_maps, cost_bps=7.5, holding_threshold=1e-8)
    assert np.max(np.abs(torch_result["gross_return"].detach().numpy() - gross)) < 1e-7
    assert np.max(np.abs(torch_result["turnover"].detach().numpy() - turnover)) < 1e-7
    assert np.max(np.abs(torch_result["cost_7_5bps"].detach().numpy() - cost)) < 1e-9
    assert np.max(np.abs(torch_result["net_return"].detach().numpy() - net)) < 1e-7
    assert np.max(np.abs(formal["gross_returns"] - gross)) < 1e-12
    assert np.max(np.abs(formal["turnovers"] - turnover)) < 1e-12


def test_forced_exit_and_new_entry_are_in_turnover() -> None:
    assets = [["a", "b"], ["b", "c"]]
    weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    returns = torch.tensor([[0.1, 0.0], [0.0, 0.0]], dtype=torch.float32)
    mask = torch.ones_like(weights, dtype=torch.bool)
    result = account_sequence_torch(weights, assets, mask, returns, mask)
    assert float(result["turnover"][1]) > 0.4  # includes selling a and buying c


def test_cost_and_net_formula() -> None:
    assets, weights, returns, mask = _toy()
    result = account_sequence_torch(weights, assets, mask, returns, mask)
    assert torch.allclose(result["cost_7_5bps"], 2.0 * result["turnover"] * 7.5 / 10000.0)
    assert torch.allclose(result["net_return"], result["gross_return"] - result["cost_7_5bps"])


def test_previous_day_weight_has_gradient() -> None:
    assets, weights, returns, mask = _toy()
    result = account_sequence_torch(weights, assets, mask, returns, mask)
    result["net_return"][1:].mean().neg().backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()
    assert float(weights.grad[0].abs().sum()) > 0


def test_batch_boundary_equivalence() -> None:
    result = _batch_boundary_test()
    assert result["status"] == "PASS"
    assert result["max_diff"] <= 1e-7


def test_d_fixed_lambda_temperature_and_scope() -> None:
    source = (ROOT / "scripts/run_daily_experiment_d.py").read_text(encoding="utf-8")
    assert LAMBDA_PORTFOLIO == 70.21024290734199
    assert "LAMBDA_PORTFOLIO = 70.21024290734199" in source
    assert "COST_BPS = 7.5" in source
    assert '"temperature": 1.0' in source
    assert "2022" in source


def test_does_not_use_extra_turnover_penalty_or_topk() -> None:
    source = (ROOT / "scripts/run_daily_experiment_d.py").read_text(encoding="utf-8")
    assert "extra_turnover_penalty" in source
    assert "topk_or_cap" in source
