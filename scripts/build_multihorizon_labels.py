"""Build paper-style multi-horizon VWAP labels aligned to a factor panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


REQUIRED_FIELDS = ("tradeDate", "ticker", "yuanVolume", "volume", "adjFactor")
LABEL_DEFINITIONS = ("paper_formula", "legacy_t_plus_1")


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ],
        dtype=str,
    )


def _as_dates(values: np.ndarray) -> pd.DatetimeIndex:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        if array.size and int(np.nanmax(array)) > 10**12:
            return pd.DatetimeIndex(pd.to_datetime(array, unit="ns"))
        return pd.DatetimeIndex(pd.to_datetime(array.astype(str), format="%Y%m%d"))
    return pd.DatetimeIndex(pd.to_datetime(_decode(array), errors="raise"))


class MarketArrays:
    """Common row/column reader for HDF5 datasets or in-memory PKL arrays."""

    def __init__(self, values: Any, source_kind: str) -> None:
        self.values = values
        self.source_kind = source_kind
        missing = [name for name in REQUIRED_FIELDS if name not in values]
        if missing:
            raise KeyError(f"Market source is missing fields: {missing}")
        self.dates = _as_dates(np.asarray(values["tradeDate"][:]))
        self.assets = _decode(np.asarray(values["ticker"][:]))
        if not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError("Market dates must be unique and increasing")
        if len(set(self.assets.tolist())) != len(self.assets):
            raise ValueError("Market tickers must be unique")

    def matrix(self, field: str, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Read monotonic rows first and reorder columns in NumPy for h5py safety."""
        output = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
        valid = (rows >= 0) & (rows < len(self.dates))
        if valid.any():
            selected_rows = rows[valid].astype(np.int64)
            block = np.asarray(self.values[field][selected_rows, :], dtype=np.float64)
            output[valid] = block[:, columns]
        return output


def _pickle_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read a cached market PKL represented as an array mapping or long DataFrame."""
    loaded = pd.read_pickle(path)
    if isinstance(loaded, dict):
        return {str(key): np.asarray(value) for key, value in loaded.items()}
    if not isinstance(loaded, pd.DataFrame):
        raise TypeError("Market PKL must contain a dict of arrays or a pandas DataFrame")
    required_columns = {
        "tradeDate",
        "ticker",
        "yuanVolume",
        "volume",
        "adjFactor",
    }
    missing = required_columns - set(loaded.columns)
    if missing:
        raise ValueError(f"Long market PKL is missing columns: {sorted(missing)}")
    frame = loaded.copy()
    frame["tradeDate"] = pd.to_datetime(frame["tradeDate"], errors="raise")
    frame["ticker"] = frame["ticker"].astype(str)
    dates = pd.DatetimeIndex(frame["tradeDate"]).sort_values().unique()
    assets = np.sort(frame["ticker"].astype(str).unique())
    arrays: dict[str, np.ndarray] = {
        "tradeDate": np.asarray(pd.DatetimeIndex(dates).strftime("%Y-%m-%d")),
        "ticker": np.asarray(assets),
    }
    for field in ("yuanVolume", "volume", "adjFactor", "zz500_weight"):
        if field in frame.columns:
            arrays[field] = (
                frame.pivot(index="tradeDate", columns="ticker", values=field)
                .reindex(index=pd.DatetimeIndex(dates), columns=assets)
                .to_numpy()
            )
    return arrays


def _adjusted_vwap(
    market: MarketArrays, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    yuan = market.matrix("yuanVolume", rows, columns)
    volume = market.matrix("volume", rows, columns)
    adjustment = market.matrix("adjFactor", rows, columns)
    output = np.divide(
        yuan,
        volume,
        out=np.full_like(yuan, np.nan),
        where=np.isfinite(volume) & (volume > 0),
    )
    output *= adjustment
    output[~np.isfinite(output) | (output <= 0)] = np.nan
    return output


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    panel_path = Path(args.panel)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    panel_dates = _as_dates(np.load(panel_path / "dates.npy", allow_pickle=False))
    panel_assets = _decode(np.load(panel_path / "asset_ids.npy", allow_pickle=False))
    horizons = tuple(int(value) for value in args.horizons)
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("Horizons must be positive integers")
    label_definition = getattr(args, "label_definition", "paper_formula")
    if label_definition not in LABEL_DEFINITIONS:
        raise ValueError(
            f"label_definition must be one of {LABEL_DEFINITIONS}, got {label_definition!r}"
        )

    pickle_path = Path(args.market_pkl) if args.market_pkl else None
    h5_handle: h5py.File | None = None
    if pickle_path is not None and pickle_path.is_file():
        market = MarketArrays(_pickle_arrays(pickle_path), "pkl")
        market_source = pickle_path
    else:
        market_source = Path(args.market_h5)
        h5_handle = h5py.File(market_source, "r")
        market = MarketArrays(h5_handle, "h5")

    try:
        market_date_position = {date: index for index, date in enumerate(market.dates)}
        decision_rows = np.asarray(
            [market_date_position.get(date, -1) for date in panel_dates], dtype=np.int64
        )
        if np.any(decision_rows < 0):
            missing = panel_dates[decision_rows < 0]
            raise ValueError(f"Market source misses panel dates, beginning with {missing[0]}")
        market_asset_position = {asset: index for index, asset in enumerate(market.assets)}
        asset_columns = np.asarray(
            [market_asset_position.get(asset, -1) for asset in panel_assets], dtype=np.int64
        )
        if np.any(asset_columns < 0):
            missing_assets = panel_assets[asset_columns < 0]
            raise ValueError(
                f"Market source misses {len(missing_assets)} panel assets; first={missing_assets[0]}"
            )

        horizon_suffix = "_".join(str(value) for value in horizons)
        returns_path = output_path / f"forward_returns_{horizon_suffix}.npy"
        benchmark_path = output_path / f"benchmark_returns_{horizon_suffix}.npy"
        execution_path = output_path / "execution_1d_return.npy"
        result = np.lib.format.open_memmap(
            returns_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(panel_dates), len(horizons), len(panel_assets)),
        )
        result[:] = np.nan
        benchmark = np.lib.format.open_memmap(
            benchmark_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(panel_dates), len(horizons)),
        )
        benchmark[:] = np.nan
        execution = np.lib.format.open_memmap(
            execution_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(panel_dates), len(panel_assets)),
        )
        execution[:] = np.nan
        has_benchmark = "zz500_weight" in market.values

        for start in range(0, len(panel_dates), int(args.block_dates)):
            stop = min(start + int(args.block_dates), len(panel_dates))
            base_rows = decision_rows[start:stop]
            entry_rows = base_rows + 1
            entry_vwap = _adjusted_vwap(market, entry_rows, asset_columns)
            execution_exit_vwap = _adjusted_vwap(market, base_rows + 2, asset_columns)
            with np.errstate(divide="ignore", invalid="ignore"):
                execution_label = execution_exit_vwap / entry_vwap - 1.0
            execution_label[~np.isfinite(execution_label)] = np.nan
            execution[start:stop] = execution_label.astype(np.float32)
            weights = (
                market.matrix("zz500_weight", entry_rows, asset_columns)
                if has_benchmark
                else None
            )
            for horizon_index, horizon in enumerate(horizons):
                # The paper formula is VWAP[t+h] / VWAP[t+1] - 1.  The legacy
                # project version used one additional exit day and is retained
                # only as an explicit sensitivity definition.
                exit_offset = horizon if label_definition == "paper_formula" else horizon + 1
                exit_rows = base_rows + exit_offset
                exit_vwap = _adjusted_vwap(market, exit_rows, asset_columns)
                with np.errstate(divide="ignore", invalid="ignore"):
                    label = exit_vwap / entry_vwap - 1.0
                label[~np.isfinite(label)] = np.nan
                result[start:stop, horizon_index] = label.astype(np.float32)
                if weights is not None:
                    valid = np.isfinite(label) & np.isfinite(weights) & (weights > 0)
                    safe_weight = np.where(valid, weights, 0.0)
                    denominator = safe_weight.sum(axis=1)
                    numerator = np.where(valid, safe_weight * label, 0.0).sum(axis=1)
                    benchmark[start:stop, horizon_index] = np.divide(
                        numerator,
                        denominator,
                        out=np.full(len(base_rows), np.nan),
                        where=denominator > 0,
                    ).astype(np.float32)
            print(f"labels {start}:{stop}/{len(panel_dates)}", flush=True)
        result.flush()
        benchmark.flush()
        execution.flush()
        finite_fraction = np.isfinite(result).mean(axis=(0, 2)).tolist()
        execution_finite_fraction = float(np.isfinite(execution).mean())
        del result, benchmark, execution
    finally:
        if h5_handle is not None:
            h5_handle.close()

    metadata = {
        "panel": str(panel_path),
        "market_source": str(market_source),
        "market_source_kind": market.source_kind,
        "dates": [str(panel_dates[0].date()), str(panel_dates[-1].date())],
        "assets": int(len(panel_assets)),
        "horizons": list(horizons),
        "entry_offset": 1,
        "label_definition": label_definition,
        "exit_offsets": [
            horizon if label_definition == "paper_formula" else horizon + 1
            for horizon in horizons
        ],
        "definition": (
            "adjusted VWAP[t+h] / adjusted VWAP[t+1] - 1"
            if label_definition == "paper_formula"
            else "adjusted VWAP[t+h+1] / adjusted VWAP[t+1] - 1"
        ),
        "finite_fraction_by_horizon": finite_fraction,
        "execution_1d_return": {
            "path": str(execution_path),
            "entry_offset": 1,
            "exit_offset": 2,
            "definition": "adjusted VWAP[t+2] / adjusted VWAP[t+1] - 1",
            "finite_fraction": execution_finite_fraction,
        },
    }
    (output_path / "multihorizon_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return returns_path, benchmark_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--market-h5", required=True)
    parser.add_argument("--market-pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 15, 20])
    parser.add_argument("--block-dates", type=int, default=64)
    parser.add_argument(
        "--label-definition",
        choices=LABEL_DEFINITIONS,
        default="paper_formula",
        help="paper formula uses VWAP[t+h]/VWAP[t+1]; legacy adds one exit day",
    )
    arguments = parser.parse_args()
    if arguments.block_dates < 1:
        parser.error("--block-dates must be positive")
    paths = build(arguments)
    print(json.dumps({
        "returns": str(paths[0]),
        "benchmark": str(paths[1]),
        "execution_1d_return": str(Path(arguments.output) / "execution_1d_return.npy"),
    }))


if __name__ == "__main__":
    main()
