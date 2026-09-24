"""Fast protocol tests for the fixed-64 annual walk-forward framework."""

from __future__ import annotations

from scripts.run_fixed64_walk_forward import FOLDS, _synthetic_boundary_test


def test_fixed_annual_fold_calendar_is_non_overlapping() -> None:
    for spec in FOLDS.values():
        assert spec["train_start"] <= spec["train_end"] < spec["validation_start"] <= spec["validation_end"] < spec["oos_start"]


def test_synthetic_switch_turnover_is_not_free() -> None:
    result = _synthetic_boundary_test()
    assert result["status"] == "PASS"
    assert result["switch_turnover"] > 0.0
    assert abs(result["switch_turnover"] - result["expected_turnover"]) < 1e-12
    assert result["separate_year_incorrect_first_turnover"] == 0.0
