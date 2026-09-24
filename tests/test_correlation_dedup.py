from __future__ import annotations

import numpy as np
import pandas as pd

from e2eai.data.correlation_dedup import (
    CorrelationResult,
    greedy_ic_then_correlation_dedup,
    train_only_ic_then_correlation_dedup,
)


def _ranked(names: list[str]) -> pd.DataFrame:
    values = np.linspace(0.10, 0.10 - 0.01 * (len(names) - 1), len(names))
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(names) + 1),
            "factor_name": names,
            "train_mean_rankic": values,
            "abs_train_mean_rankic": values,
        }
    )


def test_greedy_filter_rejects_positive_negative_and_boundary_correlations() -> None:
    names = ["A", "B", "C", "D", "E", "F", "G"]
    pairs = {
        tuple(sorted(("A", "B"))): 0.95,
        tuple(sorted(("A", "C"))): 0.20,
        tuple(sorted(("A", "D"))): -0.90,
        tuple(sorted(("A", "E"))): 0.79,
        tuple(sorted(("C", "E"))): 0.10,
        tuple(sorted(("A", "F"))): 0.80,
        tuple(sorted(("C", "F"))): 0.10,
        tuple(sorted(("E", "F"))): 0.10,
    }

    def correlation(left: str, right: str) -> CorrelationResult:
        value = pairs.get(tuple(sorted((left, right))), 0.10)
        return CorrelationResult(value, 120, 50, True)

    audit, selected = greedy_ic_then_correlation_dedup(
        _ranked(names), correlation, top_k=4, correlation_threshold=0.80
    )
    assert selected == ["A", "C", "E", "G"]
    by_name = audit.set_index("factor_name")
    assert by_name.loc["B", "rejection_reason"] == "correlation_threshold"
    assert by_name.loc["D", "rejection_reason"] == "correlation_threshold"
    assert by_name.loc["D", "signed_corr_with_most_correlated_factor"] == -0.90
    assert by_name.loc["E", "accepted"]
    assert by_name.loc["F", "rejection_reason"] == "correlation_threshold"
    assert by_name.loc["F", "max_abs_corr_to_selected_at_decision"] == 0.80


def test_corr_filter_uses_train_dates_only() -> None:
    names = ["A", "B", "C"]
    train = np.asarray([True, True, True, False, False])
    rankic = np.asarray(
        [
            [0.10, 0.09, 0.08],
            [0.10, 0.09, 0.08],
            [0.10, 0.09, 0.08],
            [-0.99, 0.99, -0.99],
            [-0.99, 0.99, -0.99],
        ]
    )
    base = np.ones((5, 6), dtype=bool)
    values = np.empty((5, 6, 3), dtype=float)
    values[:, :, 0] = np.arange(6)
    values[:, :, 1] = np.arange(6)  # B duplicates A and must be filtered.
    values[:, :, 2] = np.asarray([0, 2, 4, 1, 5, 3])
    baseline, selected = train_only_ic_then_correlation_dedup(
        names,
        rankic,
        values,
        base,
        train,
        top_k=2,
        correlation_threshold=0.80,
        min_common_stocks=5,
        min_valid_dates=3,
    )
    perturbed_values = values.copy()
    perturbed_values[~train, :, 1] = np.arange(6)[::-1]
    perturbed_rankic = rankic.copy()
    perturbed_rankic[~train] *= -100.0
    repeated, repeated_selected = train_only_ic_then_correlation_dedup(
        names,
        perturbed_rankic,
        perturbed_values,
        base,
        train,
        top_k=2,
        correlation_threshold=0.80,
        min_common_stocks=5,
        min_valid_dates=3,
    )
    assert selected == ["A", "C"]
    assert repeated_selected == selected
    assert baseline["accepted"].tolist() == repeated["accepted"].tolist()
    assert baseline["rejection_reason"].fillna("<none>").tolist() == repeated[
        "rejection_reason"
    ].fillna("<none>").tolist()
