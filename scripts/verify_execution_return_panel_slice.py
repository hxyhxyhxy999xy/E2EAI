"""Verify that a panel-sized execution-return copy equals the verified full source prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-source",
        type=Path,
        default=Path("/cloud/E2EAI/data/execution_returns/execution_1d_return_verified.npy"),
    )
    parser.add_argument(
        "--panel-copy",
        type=Path,
        default=Path(
            "/cloud/E2EAI/data/execution_returns/execution_1d_return_verified_2018_2021.npy"
        ),
    )
    parser.add_argument(
        "--dates",
        type=Path,
        default=Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08_panel/dates.npy"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/daily_strategy_diagnostics/execution_return_panel_slice_verification.json"),
    )
    args = parser.parse_args()
    source = np.load(args.full_source, mmap_mode="r", allow_pickle=False)
    panel = np.load(args.panel_copy, mmap_mode="r", allow_pickle=False)
    dates = np.load(args.dates, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(dates), source.shape[1])
    if panel.shape != expected_shape or panel.dtype != source.dtype:
        raise RuntimeError("Panel-sized execution return has incompatible shape or dtype")
    generator = np.random.default_rng(42)
    rows = generator.integers(0, panel.shape[0], size=200)
    columns = generator.integers(0, panel.shape[1], size=200)
    prefix = source[: len(dates)]
    report = {
        "full_source": str(args.full_source),
        "panel_copy": str(args.panel_copy),
        "decision_date_count": int(len(dates)),
        "full_source_sha256": _sha256(args.full_source),
        "panel_copy_sha256": _sha256(args.panel_copy),
        "full_source_shape": list(source.shape),
        "panel_copy_shape": list(panel.shape),
        "dtype": str(panel.dtype),
        "prefix_values_equal": bool(np.array_equal(prefix, panel, equal_nan=True)),
        "random_sample_count": 200,
        "random_samples_equal": bool(
            np.array_equal(prefix[rows, columns], panel[rows, columns], equal_nan=True)
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["prefix_values_equal"] or not report["random_samples_equal"]:
        raise RuntimeError("Panel-sized execution return does not match the verified source prefix")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
