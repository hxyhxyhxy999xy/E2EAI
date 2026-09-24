from __future__ import annotations

from argparse import Namespace

import h5py
import numpy as np
import pandas as pd

from scripts.build_multihorizon_labels import build


def test_builds_adjusted_vwap_to_vwap_labels(tmp_path) -> None:
    market_dates = pd.bdate_range("2020-01-02", periods=12)
    assets = np.asarray(["000001", "000002"])
    prices = np.arange(1, 13, dtype=np.float64)[:, None] * np.asarray([[1.0, 2.0]])
    volume = np.full_like(prices, 10.0)
    adjustment = np.full_like(prices, 2.0)
    yuan_volume = prices * volume / adjustment
    market_path = tmp_path / "market.h5"
    with h5py.File(market_path, "w") as store:
        store.create_dataset("tradeDate", data=market_dates.strftime("%Y%m%d").astype(int))
        store.create_dataset("ticker", data=assets.astype("S6"))
        store.create_dataset("yuanVolume", data=yuan_volume)
        store.create_dataset("volume", data=volume)
        store.create_dataset("adjFactor", data=adjustment)
        store.create_dataset("zz500_weight", data=np.full_like(prices, 0.5))

    panel_path = tmp_path / "panel"
    panel_path.mkdir()
    panel_dates = market_dates[:5]
    np.save(
        panel_path / "dates.npy",
        np.asarray(panel_dates.strftime("%Y-%m-%d"), dtype="U10"),
    )
    np.save(panel_path / "asset_ids.npy", assets)
    output_path = tmp_path / "labels"
    returns_path, benchmark_path = build(
        Namespace(
            panel=str(panel_path),
            market_h5=str(market_path),
            market_pkl=None,
            output=str(output_path),
            horizons=[1, 3, 5],
            block_dates=2,
            label_definition="legacy_t_plus_1",
        )
    )
    returns = np.load(returns_path)
    benchmark = np.load(benchmark_path)
    execution = np.load(output_path / "execution_1d_return.npy")
    assert returns.shape == (5, 3, 2)
    np.testing.assert_allclose(returns[0, 0], prices[2] / prices[1] - 1.0)
    np.testing.assert_allclose(returns[0, 1], prices[4] / prices[1] - 1.0)
    np.testing.assert_allclose(returns[0, 2], prices[6] / prices[1] - 1.0)
    np.testing.assert_allclose(benchmark[0], returns[0].mean(axis=1))
    np.testing.assert_allclose(execution[0], prices[2] / prices[1] - 1.0)
