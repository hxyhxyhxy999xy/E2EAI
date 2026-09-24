from functools import partial

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from e2eai.data.collate import collate_cross_sections
from e2eai.data.dataset import E2EAIDataset, generate_synthetic_frame, load_market_frame


HORIZONS = [3, 5, 10, 15, 20]


def test_dataset_csv_parquet_and_padding(tmp_path) -> None:
    frame = generate_synthetic_frame(
        num_dates=3,
        num_stocks=8,
        num_factors=4,
        num_industries=2,
        horizons=HORIZONS,
        seed=11,
        active_probability=0.75,
    )
    csv_path = tmp_path / "market.csv"
    parquet_path = tmp_path / "market.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    csv_frame = load_market_frame(csv_path)
    parquet_frame = load_market_frame(parquet_path)
    assert len(csv_frame) == len(parquet_frame) == len(frame)

    dataset = E2EAIDataset(csv_frame, HORIZONS)
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=partial(collate_cross_sections, universe_graph_mode="all_valid"),
    )
    batch = next(iter(loader))
    assert batch["raw_factors"].shape[:2] == batch["asset_mask"].shape
    assert batch["forward_returns"].shape[1] == len(HORIZONS)
    assert batch["industry_adjacency"].dtype == torch.bool
    assert batch["universe_adjacency"].dtype == torch.bool
    assert not torch.isnan(batch["raw_factors"]).any()


def test_missing_return_has_separate_mask() -> None:
    frame = generate_synthetic_frame(2, 6, 3, 2, HORIZONS, seed=4)
    frame.loc[frame.index[0], "forward_return_10"] = float("nan")
    dataset = E2EAIDataset(frame, HORIZONS)
    sample = dataset[0]
    assert not sample["return_mask"][2, 0]
    assert sample["forward_returns"][2, 0] == 0.0


def test_duplicate_date_asset_rejected() -> None:
    frame = generate_synthetic_frame(2, 5, 3, 2, HORIZONS, seed=2)
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        E2EAIDataset(duplicate, HORIZONS)

