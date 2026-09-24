"""Validation and inference for the canonical long-form data schema."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def infer_factor_columns(
    frame: pd.DataFrame,
    configured: Sequence[str] | None = None,
    prefix: str = "factor_",
) -> list[str]:
    """Return configured factors or deterministically infer prefix matches."""
    columns = list(configured or [])
    if not columns:
        columns = sorted(str(column) for column in frame.columns if str(column).startswith(prefix))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing factor columns: {missing}")
    if not columns:
        raise ValueError(f"No factor columns found with prefix '{prefix}'")
    return columns


def return_columns(horizons: Sequence[int], prefix: str = "forward_return_") -> list[str]:
    """Construct ordered forward-return column names."""
    return [f"{prefix}{horizon}" for horizon in horizons]


def benchmark_columns(horizons: Sequence[int], prefix: str = "benchmark_return_") -> list[str]:
    """Construct ordered optional benchmark-return column names."""
    return [f"{prefix}{horizon}" for horizon in horizons]


def validate_long_frame(
    frame: pd.DataFrame,
    date_column: str,
    asset_column: str,
    industry_column: str,
    factors: Sequence[str],
    returns: Sequence[str],
) -> None:
    """Validate identifiers, feature/label columns, and unique date-assets."""
    required = [date_column, asset_column, industry_column, *factors, *returns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    duplicates = frame.duplicated([date_column, asset_column])
    if duplicates.any():
        sample = frame.loc[duplicates, [date_column, asset_column]].head().to_dict("records")
        raise ValueError(f"Duplicate date/asset rows are not allowed; examples: {sample}")

