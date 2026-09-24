from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from e2eai.config import ExperimentConfig
from e2eai.data.loaders import build_dataloaders
from e2eai.data.panel import AlphaPanelDataset, load_alpha_panel
from e2eai.workflows import load_configured_data


def _arrays() -> dict[str, np.ndarray]:
    dates = np.asarray([f"2024-01-{day:02d}" for day in range(2, 10)])
    assets = np.asarray(["000001", "000002", "000003", "000004", "000005"])
    alpha = np.arange(8 * 3 * 5, dtype=np.float32).reshape(8, 3, 5)
    returns = np.arange(8 * 5, dtype=np.float32).reshape(8, 5) / 1000.0
    eligible = np.ones((8, 5), dtype=bool)
    eligible[0, 1] = False
    return {
        "dates": dates,
        "asset_ids": assets,
        "alpha_names": np.asarray(["value", "quality", "momentum"]),
        "alpha_tensor": alpha,
        "forward_returns": returns,
        "benchmark_returns": np.linspace(0.0, 0.007, 8, dtype=np.float32),
        "eligible_mask": eligible,
    }


def _config(path: Path) -> ExperimentConfig:
    config = ExperimentConfig()
    config.data.path = str(path)
    config.data.format = "alpha_panel"
    config.data.max_assets_per_date = 4
    config.data.train_ratio = 0.50
    config.data.validation_ratio = 0.25
    config.data.test_ratio = 0.25
    config.model.num_factors = 3
    config.model.horizons = [1]
    config.training.batch_dates = 2
    config.validate()
    return config


def test_npy_directory_is_memory_mapped_and_transposed(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel"
    panel_path.mkdir()
    for name, values in _arrays().items():
        np.save(panel_path / f"{name}.npy", values, allow_pickle=False)
    industry_path = tmp_path / "industry.npz"
    np.savez_compressed(
        industry_path,
        index=np.asarray(["2024-01-01", "2024-01-05"]),
        columns=np.asarray(["000001", "000003", "000004", "000005"]),
        values=np.asarray([[10, 20, 20, 30], [11, 21, 21, 31]], dtype=np.float32),
    )
    config = _config(panel_path)
    config.data.industry_npz_path = str(industry_path)
    panel = load_alpha_panel(panel_path, config.data)
    assert isinstance(panel.alpha_tensor, np.memmap)
    dataset = AlphaPanelDataset(panel, config.model.horizons, config.data)
    sample = dataset[0]
    assert sample["raw_factors"].shape == (4, 3)
    expected_positions = np.asarray([0, 2, 3, 4])
    expected = _arrays()["alpha_tensor"][0, :, expected_positions]
    np.testing.assert_allclose(sample["raw_factors"].numpy(), expected)
    assert sample["asset_ids"] == ["000001", "000003", "000004", "000005"]
    assert sample["industry_ids"].tolist() == [10, 20, 20, 30]


def test_npz_auto_detection_and_dataloader(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.npz"
    np.savez_compressed(panel_path, **_arrays())
    config = _config(panel_path)
    config.data.format = "auto"
    source, synthetic = load_configured_data(config)
    assert not synthetic
    loaders = build_dataloaders(config, source)
    batch = next(iter(loaders.train))
    assert batch["raw_factors"].shape == (2, 4, 3)
    assert batch["forward_returns"].shape == (2, 1, 4)
    assert batch["industry_ids"].eq(-1).all()


def test_single_label_requires_one_model_horizon(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.npz"
    np.savez_compressed(panel_path, **_arrays())
    config = _config(panel_path)
    config.model.horizons = [3, 5]
    panel = load_alpha_panel(panel_path, config.data)
    with pytest.raises(ValueError, match="one forward-return label"):
        AlphaPanelDataset(panel, config.model.horizons, config.data)


def test_explicit_broadcast_policy(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.npz"
    np.savez_compressed(panel_path, **_arrays())
    config = _config(panel_path)
    config.model.horizons = [3, 5]
    config.data.single_return_policy = "broadcast"
    sample = AlphaPanelDataset(
        load_alpha_panel(panel_path, config.data), config.model.horizons, config.data
    )[0]
    assert sample["forward_returns"].shape == (2, 4)
    np.testing.assert_allclose(
        sample["forward_returns"][0].numpy(), sample["forward_returns"][1].numpy()
    )


def test_external_multihorizon_return_override(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel"
    panel_path.mkdir()
    for name, values in _arrays().items():
        np.save(panel_path / f"{name}.npy", values, allow_pickle=False)
    returns = np.stack([_arrays()["forward_returns"] * scale for scale in (1, 2, 3)])
    returns = np.moveaxis(returns, 0, 1)
    benchmark = np.arange(8 * 3, dtype=np.float32).reshape(8, 3) / 1000.0
    return_path = tmp_path / "multi_returns.npy"
    benchmark_path = tmp_path / "multi_benchmark.npy"
    np.save(return_path, returns)
    np.save(benchmark_path, benchmark)
    config = _config(panel_path)
    config.model.horizons = [3, 5, 10]
    config.data.panel_return_horizons = [3, 5, 10]
    config.data.return_path = str(return_path)
    config.data.benchmark_path = str(benchmark_path)
    panel = load_alpha_panel(panel_path, config.data)
    sample = AlphaPanelDataset(panel, config.model.horizons, config.data)[0]
    assert sample["forward_returns"].shape == (3, 4)
    np.testing.assert_allclose(sample["benchmark_returns"].numpy(), benchmark[0])


def test_return_horizon_metadata_must_match_model(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.npz"
    arrays = _arrays()
    arrays["forward_returns"] = np.repeat(
        arrays["forward_returns"][:, None, :], 2, axis=1
    )
    np.savez_compressed(panel_path, **arrays)
    config = _config(panel_path)
    config.model.horizons = [3, 5]
    config.data.panel_return_horizons = [5, 3]
    panel = load_alpha_panel(panel_path, config.data)
    with pytest.raises(ValueError, match="must exactly match"):
        AlphaPanelDataset(panel, config.model.horizons, config.data)
