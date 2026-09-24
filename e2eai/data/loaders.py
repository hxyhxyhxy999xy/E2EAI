"""Configuration-driven chronological DataLoader construction."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Protocol

import pandas as pd
from torch.utils.data import DataLoader, Subset

from e2eai.config import ExperimentConfig
from e2eai.data.collate import collate_cross_sections
from e2eai.data.dataset import E2EAIDataset
from e2eai.data.panel import AlphaPanel, AlphaPanelDataset
from e2eai.data.splits import ChronologicalBatchSampler


class CrossSectionDataset(Protocol):
    dates: tuple[pd.Timestamp, ...]
    factor_columns: tuple[str, ...] | list[str]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int): ...


class LoaderBundle(NamedTuple):
    """Dataset and chronological train/validation/test loaders."""

    dataset: E2EAIDataset | AlphaPanelDataset
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def _bounded_indices(
    dates: tuple[pd.Timestamp, ...], start: str | None, end: str | None
) -> list[int]:
    lower = pd.Timestamp.min if start is None else pd.Timestamp(start)
    upper = pd.Timestamp.max if end is None else pd.Timestamp(end)
    return [index for index, date in enumerate(dates) if lower <= date <= upper]


def split_indices(
    config: ExperimentConfig, dataset: CrossSectionDataset
) -> tuple[list[int], list[int], list[int]]:
    """Return chronological splits, optionally purged by label exit dates."""
    train, validation, test, _ = _split_indices_with_metadata(config, dataset)
    return train, validation, test


def _split_indices_with_metadata(
    config: ExperimentConfig, dataset: CrossSectionDataset
) -> tuple[list[int], list[int], list[int], dict[str, object]]:
    """Build splits and remove observations whose labels cross split boundaries."""
    data = config.data
    explicit = any(
        value is not None
        for value in (
            data.train_start,
            data.train_end,
            data.validation_start,
            data.validation_end,
            data.test_start,
            data.test_end,
        )
    )
    if explicit:
        train = _bounded_indices(dataset.dates, data.train_start, data.train_end)
        validation = _bounded_indices(dataset.dates, data.validation_start, data.validation_end)
        # Validation-only daily cores intentionally leave test bounds unset;
        # an unset test range must mean an empty test split, not "all dates".
        test = (
            []
            if data.daily_validation_only
            and data.test_start is None
            and data.test_end is None
            else _bounded_indices(dataset.dates, data.test_start, data.test_end)
        )
        overlap = (set(train) & set(validation)) | (set(train) & set(test)) | (
            set(validation) & set(test)
        )
        if overlap:
            raise ValueError(f"Explicit chronological splits overlap at indices: {sorted(overlap)}")
    else:
        total = len(dataset)
        train_end = int(total * data.train_ratio)
        validation_end = train_end + int(total * data.validation_ratio)
        train = list(range(0, train_end))
        validation = list(range(train_end, validation_end))
        test = list(range(validation_end, total))
    raw_train, raw_validation, raw_test = list(train), list(validation), list(test)
    validation_only = bool(data.daily_validation_only)
    if validation_only and config.experiment_line != "daily_strategy":
        raise ValueError("daily_validation_only is supported only for daily_strategy")
    offsets = list(data.label_exit_offsets)
    if not offsets:
        offsets = list(data.panel_return_horizons or config.model.horizons)
        if data.label_definition == "legacy_t_plus_1":
            offsets = [int(value) + 1 for value in offsets]
    purge_metadata: dict[str, object] = {
        "purge_label_overlap": bool(data.purge_label_overlap),
        "label_definition": data.label_definition,
        "label_exit_offsets": [int(value) for value in offsets],
        "raw_counts": {
            "train": len(raw_train),
            "validation": len(raw_validation),
            "test": len(raw_test),
        },
        "purged_counts": {"train": 0, "validation": 0},
    }
    if data.purge_label_overlap:
        if not offsets or any(int(value) < 1 for value in offsets):
            raise ValueError("Label exit offsets are required for overlap purging")
        max_offset = max(int(value) for value in offsets)
        purge_metadata["max_exit_offset"] = max_offset
        if raw_validation:
            validation_boundary = min(raw_validation)
            train = [
                index
                for index in raw_train
                if index + max_offset < validation_boundary
            ]
        if raw_test:
            test_boundary = min(raw_test)
            validation = [
                index
                for index in raw_validation
                if index + max_offset < test_boundary
            ]
        elif validation_only:
            # The panel deliberately stops before the test period.  Drop the
            # final decisions whose longest label would leave this panel, so
            # diagnostics and early stopping never consume 2022 returns.
            validation = [
                index
                for index in raw_validation
                if index + max_offset < len(dataset.dates)
            ]
        purge_metadata["purged_counts"] = {
            "train": len(raw_train) - len(train),
            "validation": len(raw_validation) - len(validation),
        }
        purge_metadata["boundary_dates"] = {
            "validation": str(dataset.dates[min(raw_validation)].date()) if raw_validation else None,
            "test": str(dataset.dates[min(raw_test)].date()) if raw_test else None,
            "validation_only_panel_end": (
                str(dataset.dates[-1].date()) if validation_only and not raw_test else None
            ),
        }
    if not train or not validation or (not test and not validation_only):
        raise ValueError("Train and validation splits, plus test unless validation-only, must contain dates")
    if max(train) >= min(validation) or (test and max(validation) >= min(test)):
        raise ValueError("Splits must be strictly chronological: train, validation, then test")
    purge_metadata["used_counts"] = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }
    purge_metadata["used_date_ranges"] = {
        "train": [str(dataset.dates[min(train)].date()), str(dataset.dates[max(train)].date())],
        "validation": [
            str(dataset.dates[min(validation)].date()),
            str(dataset.dates[max(validation)].date()),
        ],
        "test": (
            [str(dataset.dates[min(test)].date()), str(dataset.dates[max(test)].date())]
            if test
            else None
        ),
    }
    return train, validation, test, purge_metadata


def build_dataloaders(
    config: ExperimentConfig, source: pd.DataFrame | AlphaPanel
) -> LoaderBundle:
    """Create chronological loaders from long-form data or an AlphaGAT panel."""
    if isinstance(source, AlphaPanel):
        dataset: E2EAIDataset | AlphaPanelDataset = AlphaPanelDataset(
            source,
            config.model.horizons,
            config.data,
            execution_horizon=config.model.execution_horizon,
        )
    elif isinstance(source, pd.DataFrame):
        dataset = E2EAIDataset(
            source,
            config.model.horizons,
            config.data.factor_columns,
            date_column=config.data.date_column,
            asset_column=config.data.asset_column,
            industry_column=config.data.industry_column,
            factor_prefix=config.data.factor_prefix,
            return_prefix=config.data.return_prefix,
            benchmark_prefix=config.data.benchmark_prefix,
            execution_horizon=config.model.execution_horizon,
            execution_return_column=config.data.execution_return_column,
            missing_factor_strategy=config.data.missing_factor_strategy,
        )
    else:
        raise TypeError(f"Unsupported data source: {type(source).__name__}")
    if len(dataset.factor_columns) != config.model.num_factors:
        raise ValueError(
            f"Configured model expects {config.model.num_factors} factors but data has "
            f"{len(dataset.factor_columns)}"
        )
    train_indices, validation_indices, test_indices, split_metadata = _split_indices_with_metadata(
        config, dataset
    )
    dataset.split_metadata = split_metadata
    collate = partial(
        collate_cross_sections,
        universe_graph_mode=config.data.universe_graph_mode,
        unknown_industry_self_only=config.data.unknown_industry_self_only,
    )

    def loader(indices: list[int]) -> DataLoader:
        subset = Subset(dataset, indices)
        sampler = ChronologicalBatchSampler(
            list(range(len(subset))), config.training.batch_dates, drop_last=False
        )
        return DataLoader(
            subset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=config.training.num_workers,
        )

    return LoaderBundle(dataset, loader(train_indices), loader(validation_indices), loader(test_indices))
