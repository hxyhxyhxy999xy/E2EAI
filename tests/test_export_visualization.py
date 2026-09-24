from pathlib import Path

import pandas as pd

from e2eai.models.e2eai import E2EAIModel
from e2eai.workflows import save_first_batch_artifacts


def test_exports_and_all_required_plots(tiny_config, tiny_loaders) -> None:
    model = E2EAIModel(tiny_config.model)
    batch = next(iter(tiny_loaders.test))
    artifacts = save_first_batch_artifacts(
        tiny_config.logging.output_dir,
        model=model,
        batch=batch,
        device=next(model.parameters()).device,
        horizons=tiny_config.model.horizons,
        factor_names=tiny_loaders.dataset.factor_columns,
        factor_groups={},
        history=[{"train_total_loss": 1.0, "selected_factor_count": 2.0, "selected_stock_count": 4.0, "max_weight": 0.25}],
        cap_mode="capped_simplex",
    )
    predictions = pd.read_parquet(artifacts["predictions_path"])
    interpretation = pd.read_parquet(artifacts["interpretation_path"])
    assert {
        "date", "asset_id", "horizon", "deep_factor", "deep_factor_approx",
        "portfolio_attention", "selected", "portfolio_weight",
    }.issubset(predictions.columns)
    assert {"factor_name", "factor_attention", "direction", "signed_contribution"}.issubset(
        interpretation.columns
    )
    assert len(artifacts["figures"]) == 10
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in artifacts["figures"])

