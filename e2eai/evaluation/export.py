"""Parquet exports for portfolio decisions and factor interpretation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from e2eai.models.e2eai import E2EAIOutput


def export_predictions(
    path: str | Path,
    output: "E2EAIOutput",
    dates: Sequence[object],
    asset_ids: Sequence[Sequence[str]],
    horizons: Sequence[int],
) -> Path:
    """Export stock-level model outputs in canonical long format."""
    rows: list[dict[str, object]] = []
    detached = output.detached_cpu()
    for batch_index, date in enumerate(dates):
        for horizon_index, horizon in enumerate(horizons):
            for stock_index, asset_id in enumerate(asset_ids[batch_index]):
                rows.append(
                    {
                        "date": date,
                        "asset_id": asset_id,
                        "horizon": int(horizon),
                        "deep_factor": float(detached.deep_factors[batch_index, horizon_index, stock_index]),
                        "deep_factor_approx": float(
                            detached.deep_factor_approx[batch_index, horizon_index, stock_index]
                        ),
                        "portfolio_attention": float(
                            detached.portfolio_attention[batch_index, horizon_index, stock_index]
                        ),
                        "selected": bool(
                            detached.stock_selection_mask[batch_index, horizon_index, stock_index]
                        ),
                        "portfolio_weight": float(
                            detached.portfolio_weights[batch_index, horizon_index, stock_index]
                        ),
                    }
                )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    return target


def export_factor_interpretability(
    path: str | Path,
    output: "E2EAIOutput",
    dates: Sequence[object],
    factor_names: Sequence[str],
    horizons: Sequence[int],
) -> Path:
    """Export date/horizon/factor attention and signed contribution."""
    rows: list[dict[str, object]] = []
    detached = output.detached_cpu()
    for batch_index, date in enumerate(dates):
        for horizon_index, horizon in enumerate(horizons):
            for factor_index, factor_name in enumerate(factor_names):
                attention = float(
                    detached.factor_attention_mean[batch_index, horizon_index, factor_index]
                )
                direction = float(detached.factor_directions[horizon_index, factor_index])
                rows.append(
                    {
                        "date": date,
                        "horizon": int(horizon),
                        "factor_name": factor_name,
                        "factor_attention": attention,
                        "direction": direction,
                        "signed_contribution": attention * direction,
                    }
                )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    return target

