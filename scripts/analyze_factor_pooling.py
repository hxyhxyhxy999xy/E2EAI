"""Diagnose whether cross-sectional mean pooling is nearly degenerate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders, split_indices
from e2eai.workflows import load_configured_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source, _ = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    train, validation, test = split_indices(config, loaders.dataset)
    indices = {
        "train": train,
        "validation": validation,
        "test": test,
        "all": list(range(len(loaders.dataset))),
    }[args.split]
    daily_means: list[np.ndarray] = []
    for index in indices:
        sample = loaders.dataset[index]
        values = sample["raw_factors"].float().numpy()
        daily_means.append(np.nanmean(values, axis=0))
    matrix = np.stack(daily_means)
    rows = []
    for factor, name in enumerate(loaders.dataset.factor_columns):
        series = matrix[:, factor]
        rows.append(
            {
                "factor": str(name),
                "std_t_mean_i": float(np.nanstd(series, ddof=0)),
                "mean_t_mean_i": float(np.nanmean(series)),
                "mean_abs_t_mean_i": float(np.nanmean(np.abs(series))),
                "near_zero_fraction": float(np.mean(np.abs(series) < 1e-6)),
                "dates": int(np.isfinite(series).sum()),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("std_t_mean_i").to_csv(
        args.output, index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()

