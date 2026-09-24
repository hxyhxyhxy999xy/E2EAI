"""Build the leakage-safe CSI500 t+1 -> t+2 VWAP benchmark path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.build_multihorizon_labels import MarketArrays, _adjusted_vwap, _as_dates, _decode


def build(panel: Path, market_h5: Path, output: Path) -> tuple[Path, Path]:
    panel_dates = _as_dates(np.load(panel / "dates.npy", allow_pickle=False))
    with h5py.File(market_h5, "r") as handle:
        market = MarketArrays(handle, "h5")
        if "zz500_weight" not in handle:
            raise KeyError("Market source is missing zz500_weight")
        date_lookup = {date: index for index, date in enumerate(market.dates)}
        rows = np.asarray([date_lookup.get(date, -1) for date in panel_dates], dtype=np.int64)
        if np.any(rows < 0):
            missing = panel_dates[rows < 0]
            raise ValueError(f"Market source misses panel dates beginning with {missing[0]}")
        weights = np.asarray(handle["zz500_weight"][rows], dtype=np.float64)
        columns = np.arange(len(market.assets), dtype=np.int64)
        result = np.full(len(panel_dates), np.nan, dtype=np.float32)
        finite_coverage = np.full(len(panel_dates), np.nan, dtype=np.float32)
        for start in range(0, len(rows), 64):
            stop = min(start + 64, len(rows))
            base = rows[start:stop]
            entry = _adjusted_vwap(market, base + 1, columns)
            exit_ = _adjusted_vwap(market, base + 2, columns)
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = exit_ / entry - 1.0
            valid = np.isfinite(returns) & np.isfinite(weights[start:stop]) & (weights[start:stop] > 0)
            observed = np.where(valid, weights[start:stop], 0.0).sum(axis=1)
            result[start:stop] = np.divide(
                np.where(valid, weights[start:stop] * np.nan_to_num(returns), 0.0).sum(axis=1),
                observed,
                out=np.full(stop - start, np.nan),
                where=observed > 0,
            ).astype(np.float32)
            finite_coverage[start:stop] = np.divide(
                valid.sum(axis=1),
                (weights[start:stop] > 0).sum(axis=1),
                out=np.full(stop - start, np.nan),
                where=(weights[start:stop] > 0).sum(axis=1) > 0,
            ).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result)
    metadata_path = output.with_name(output.stem + "_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "path": str(output),
                "source_market": str(market_h5),
                "source_field": "zz500_weight",
                "dates": [str(panel_dates[0].date()), str(panel_dates[-1].date())],
                "shape": list(result.shape),
                "decision_date_definition": "decision date t uses decision-date CSI500 weights",
                "return_definition": "adjusted VWAP[t+2] / adjusted VWAP[t+1] - 1",
                "entry_offset": 1,
                "exit_offset": 2,
                "finite_fraction": float(np.isfinite(result).mean()),
                "coverage_mean": float(np.nanmean(finite_coverage)),
                "coverage_min": float(np.nanmin(finite_coverage)),
                "coverage_max": float(np.nanmax(finite_coverage)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--market-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in build(args.panel, args.market_h5, args.output):
        print(path)


if __name__ == "__main__":
    main()
