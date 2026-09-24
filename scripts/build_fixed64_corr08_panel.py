"""Build the fixed-64 corr08 daily panel through the latest legal execution date.

This is intentionally *not* a factor-selection script.  It consumes the
already frozen factor list and direction map, applies the exact same daily
preprocessing as the accepted train-only corr08 panel, and extends the arrays
through the available market calendar.  It also proves that the overlapping
2018--2021 panel is identical before any walk-forward model is allowed to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_multihorizon_labels import MarketArrays, _adjusted_vwap
from scripts.build_trainonly64_panel import _as_dates, _decode, _pricing_base_arrays, _read_aligned_factor


TOLERANCE = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_factor_list(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("factor_names") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or len(values) != 64 or not all(isinstance(item, str) for item in values):
        raise ValueError("factor_list.json must contain exactly 64 factor names")
    if len(set(values)) != 64:
        raise ValueError("fixed factor list contains duplicates")
    return values


def _load_directions(path: Path, names: list[str]) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("directions", payload) if isinstance(payload, dict) else None
    if not isinstance(mapping, dict):
        raise ValueError("factor_directions.json must be a mapping")
    directions = {str(name): int(value) for name, value in mapping.items()}
    if set(directions) != set(names):
        raise ValueError("frozen directions must have exactly the same names as factor_list.json")
    if not all(value in {-1, 1} for value in directions.values()):
        raise ValueError("frozen factor directions must be -1 or +1")
    return directions


def _execution_and_benchmark(
    market_path: Path,
    dates: pd.DatetimeIndex,
    assets: np.ndarray,
    *,
    block_dates: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return VWAP[t+2]/VWAP[t+1]-1 and same-timing CSI500 benchmark."""
    with h5py.File(market_path, "r") as handle:
        market = MarketArrays(handle, "h5")
        if "zz500_weight" not in handle:
            raise KeyError("market H5 lacks zz500_weight required for CSI500 timing parity")
        date_position = {date: index for index, date in enumerate(market.dates)}
        asset_position = {asset: index for index, asset in enumerate(market.assets)}
        rows = np.asarray([date_position.get(date, -1) for date in dates], dtype=np.int64)
        columns = np.asarray([asset_position.get(asset, -1) for asset in assets], dtype=np.int64)
        if np.any(rows < 0) or np.any(columns < 0):
            raise ValueError("market axes cannot be aligned to fixed-panel axes")
        execution = np.full((len(dates), len(assets)), np.nan, dtype=np.float32)
        benchmark = np.full(len(dates), np.nan, dtype=np.float32)
        coverage = np.full(len(dates), np.nan, dtype=np.float32)
        for start in range(0, len(rows), block_dates):
            stop = min(start + block_dates, len(rows))
            base = rows[start:stop]
            entry = _adjusted_vwap(market, base + 1, columns)
            exit_ = _adjusted_vwap(market, base + 2, columns)
            with np.errstate(divide="ignore", invalid="ignore"):
                values = exit_ / entry - 1.0
            values[~np.isfinite(values)] = np.nan
            execution[start:stop] = values.astype(np.float32)
            # Benchmark membership/weights are known on decision date t.  The
            # economic return remains the same t+1-to-t+2 VWAP interval.
            weights = market.matrix("zz500_weight", base, columns)
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            denominator = np.where(valid, weights, 0.0).sum(axis=1)
            numerator = np.where(valid, weights * values, 0.0).sum(axis=1)
            benchmark[start:stop] = np.divide(
                numerator,
                denominator,
                out=np.full(stop - start, np.nan),
                where=denominator > 0,
            ).astype(np.float32)
            membership = (weights > 0).sum(axis=1)
            coverage[start:stop] = np.divide(
                valid.sum(axis=1), membership, out=np.full(stop - start, np.nan), where=membership > 0
            ).astype(np.float32)
    latest_market = str(market.dates[-1].date())
    legal = np.isfinite(benchmark)
    if not legal.any():
        raise RuntimeError("no legal CSI500 decision date has a t+1/t+2 return")
    last = int(np.flatnonzero(legal)[-1])
    metadata = {
        "definition": "adjusted VWAP[t+2] / adjusted VWAP[t+1] - 1",
        "entry_offset": 1,
        "exit_offset": 2,
        "decision_date_indexing": "t",
        "benchmark_weights_date": "t decision-date CSI500 weights",
        "raw_latest_market_date": latest_market,
        "latest_valid_decision_date": str(dates[last].date()),
        "latest_valid_t_plus_1_date": str(market.dates[rows[last] + 1].date()),
        "latest_valid_t_plus_2_date": str(market.dates[rows[last] + 2].date()),
        "benchmark_finite_coverage_mean": float(np.nanmean(coverage)),
        "benchmark_finite_coverage_min": float(np.nanmin(coverage)),
    }
    return execution, benchmark, metadata


def _process_factors(
    destination: Path,
    dates: pd.DatetimeIndex,
    assets: np.ndarray,
    names: list[str],
    candidate_dir: Path,
    base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact fixed-list version of the accepted daily fill/winsor/z-score logic."""
    count, asset_count = len(dates), len(assets)
    alpha = np.lib.format.open_memmap(
        destination / "alpha_tensor.npy", mode="w+", dtype=np.float32, shape=(count, len(names), asset_count)
    )
    factor_valid = np.lib.format.open_memmap(
        destination / "factor_valid_mask.npy", mode="w+", dtype=bool, shape=(count, len(names), asset_count)
    )
    coverage = np.zeros((count, asset_count), dtype=np.uint8)
    for factor_index, name in enumerate(names):
        path = candidate_dir / f"{name}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"fixed factor is unavailable: {path}")
        raw = _read_aligned_factor(path, dates, assets, dtype=np.float64)
        valid = np.isfinite(raw) & base
        factor_valid[:, factor_index, :] = valid
        coverage += valid.astype(np.uint8)
        processed = np.zeros_like(raw, dtype=np.float32)
        for date_index in range(count):
            valid_row = valid[date_index]
            if not valid_row.any():
                continue
            values = raw[date_index, valid_row].astype(np.float64)
            median = float(np.nanmedian(values))
            filled = np.where(np.isfinite(raw[date_index]), raw[date_index], median).astype(np.float64)
            filled = np.where(base[date_index], filled, np.nan)
            finite = np.isfinite(filled)
            if not finite.any():
                continue
            lower, upper = np.nanquantile(filled[finite], [0.01, 0.99])
            filled[finite] = np.clip(filled[finite], lower, upper)
            mean = float(np.nanmean(filled[finite]))
            std = float(np.nanstd(filled[finite]))
            if std > 1e-12:
                processed[date_index, finite] = ((filled[finite] - mean) / std).astype(np.float32)
        alpha[:, factor_index, :] = processed
        print(f"fixed_panel_factor={factor_index + 1}/{len(names)} {name}", flush=True)
    alpha.flush(); factor_valid.flush()
    del alpha, factor_valid
    return coverage, np.asarray(np.load(destination / "factor_valid_mask.npy", mmap_mode="r", allow_pickle=False))


def _array_comparison(left: np.ndarray, right: np.ndarray, *, block_dates: int) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"same_shape": False, "shape_left": list(left.shape), "shape_right": list(right.shape), "finite_mask_match": False, "mismatch_count": None, "max_abs_diff": None}
    mismatch_count = 0
    finite_mismatch = 0
    max_abs_diff = 0.0
    for start in range(0, left.shape[0], block_dates):
        stop = min(start + block_dates, left.shape[0])
        a, b = np.asarray(left[start:stop]), np.asarray(right[start:stop])
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        finite_mismatch += int(np.count_nonzero(finite_a != finite_b))
        common = finite_a & finite_b
        if common.any():
            diff = np.abs(a[common].astype(np.float64) - b[common].astype(np.float64))
            max_abs_diff = max(max_abs_diff, float(diff.max()))
            mismatch_count += int(np.count_nonzero(diff > TOLERANCE))
    return {
        "same_shape": True,
        "shape_left": list(left.shape),
        "shape_right": list(right.shape),
        "finite_mask_match": finite_mismatch == 0,
        "finite_mask_mismatch_count": finite_mismatch,
        "mismatch_count": mismatch_count,
        "max_abs_diff": max_abs_diff,
        "tolerance": TOLERANCE,
    }


def _overlap_identity(
    new_panel: Path,
    old_panel: Path,
    old_execution: Path,
    old_benchmark: Path,
    *,
    block_dates: int,
) -> dict[str, Any]:
    new_dates = _decode(np.load(new_panel / "dates.npy", allow_pickle=False))
    old_dates = _decode(np.load(old_panel / "dates.npy", allow_pickle=False))
    old_count = len(old_dates)
    new_assets = _decode(np.load(new_panel / "asset_ids.npy", allow_pickle=False))
    old_assets = _decode(np.load(old_panel / "asset_ids.npy", allow_pickle=False))
    new_names = _decode(np.load(new_panel / "alpha_names.npy", allow_pickle=False))
    old_names = _decode(np.load(old_panel / "alpha_names.npy", allow_pickle=False))
    axis = {
        "date_axis_match": bool(np.array_equal(new_dates[:old_count], old_dates)),
        "asset_axis_match": bool(np.array_equal(new_assets, old_assets)),
        "factor_axis_match": bool(np.array_equal(new_names, old_names)),
        "overlap_date_range": [str(old_dates[0]), str(old_dates[-1])],
        "overlap_date_count": old_count,
    }
    alpha = _array_comparison(
        np.load(new_panel / "alpha_tensor.npy", mmap_mode="r", allow_pickle=False)[:old_count],
        np.load(old_panel / "alpha_tensor.npy", mmap_mode="r", allow_pickle=False),
        block_dates=block_dates,
    )
    factor_valid = _array_comparison(
        np.load(new_panel / "factor_valid_mask.npy", mmap_mode="r", allow_pickle=False)[:old_count],
        np.load(old_panel / "factor_valid_mask.npy", mmap_mode="r", allow_pickle=False),
        block_dates=block_dates,
    )
    execution = _array_comparison(
        np.load(new_panel / "execution_1d_return.npy", mmap_mode="r", allow_pickle=False)[:old_count],
        np.load(old_execution, mmap_mode="r", allow_pickle=False),
        block_dates=block_dates,
    )
    benchmark = _array_comparison(
        np.load(new_panel / "csi500_execution_1d_return.npy", mmap_mode="r", allow_pickle=False)[:old_count],
        np.load(old_benchmark, mmap_mode="r", allow_pickle=False),
        block_dates=block_dates,
    )
    passed = all(axis[key] for key in ("date_axis_match", "asset_axis_match", "factor_axis_match"))
    for result in (alpha, factor_valid, execution, benchmark):
        passed = passed and result.get("same_shape", False) and result.get("finite_mask_match", False) and int(result.get("mismatch_count") or 0) == 0
    return {"status": "PASS" if passed else "FAIL", "axis": axis, "alpha_tensor": alpha, "factor_valid_mask": factor_valid, "execution_return": execution, "benchmark": benchmark}


def build(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite fixed panel output: {output}")
    names = _load_factor_list(args.factor_list)
    directions = _load_directions(args.factor_directions, names)
    with h5py.File(args.market_h5, "r") as market:
        dates = _as_dates(np.asarray(market["tradeDate"][:]))
        assets = _decode(np.asarray(market["ticker"][:]))
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("market calendar must be unique and increasing")
    old_dates = _as_dates(np.load(args.old_panel / "dates.npy", allow_pickle=False))
    old_assets = _decode(np.load(args.old_panel / "asset_ids.npy", allow_pickle=False))
    if not np.array_equal(assets[np.isin(assets, old_assets)], old_assets):
        # We deliberately choose the existing formal asset axis.  The market may contain one extra ticker.
        assets = old_assets
    elif not np.array_equal(assets, old_assets):
        assets = old_assets
    output.mkdir(parents=True, exist_ok=False)
    pricing = _pricing_base_arrays(args.market_h5, dates, assets, block_dates=args.block_dates)
    execution, benchmark, timing = _execution_and_benchmark(args.market_h5, dates, assets, block_dates=args.block_dates)
    np.save(output / "dates.npy", np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"))
    np.save(output / "asset_ids.npy", assets.astype(str))
    np.save(output / "alpha_names.npy", np.asarray(names, dtype="U"))
    np.save(output / "base_tradable_mask.npy", pricing["base_tradable"].astype(bool))
    np.save(output / "market_cap.npy", pricing["market_cap"].astype(np.float32))
    np.save(output / "free_tradable_share.npy", pricing["free_share"].astype(np.float32))
    np.save(output / "free_float_market_cap.npy", (pricing["market_cap"] * pricing["free_share"]).astype(np.float32))
    np.save(output / "execution_1d_return.npy", execution)
    # A 2-D compatibility return file; no multi-horizon labels are loaded by this experiment.
    np.save(output / "forward_returns.npy", execution)
    np.save(output / "csi500_execution_1d_return.npy", benchmark)
    coverage, _ = _process_factors(output, dates, assets, names, args.factor_dir, pricing["base_tradable"])
    eligible = pricing["base_tradable"] & (coverage >= int(np.ceil(0.90 * len(names)))) & np.isfinite(execution)
    np.save(output / "eligible_mask.npy", eligible)
    np.save(output / "normal_eligible_mask.npy", eligible)
    last = pd.Timestamp(timing["latest_valid_decision_date"])
    meta = {
        "schema": "fixed64_corr08_extended_panel_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fixed_factor_list_path": str(args.factor_list),
        "fixed_factor_directions_path": str(args.factor_directions),
        "fixed_factor_list_sha256": _sha256(args.factor_list),
        "fixed_factor_directions_sha256": _sha256(args.factor_directions),
        "fixed_factor_count": len(names),
        "factor_reselection_invoked": False,
        "correlation_selection_invoked": False,
        "factor_direction_applied_to_alpha_tensor": False,
        "factor_preprocessing": "per-date median fill, 1/99 winsorisation, cross-sectional z-score",
        "execution_return": timing,
        "panel_date_range": [str(dates[0].date()), str(dates[-1].date())],
        "latest_legal_execution_date": str(last.date()),
        "eligible_dates_through": str(last.date()),
    }
    (output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    identity = _overlap_identity(output, args.old_panel, args.old_execution, args.old_benchmark, block_dates=args.block_dates)
    (output / "overlap_identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    if identity["status"] != "PASS":
        raise RuntimeError(f"fixed-panel overlap identity failed: {identity}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-list", type=Path, default=Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08/factor_list.json"))
    parser.add_argument("--factor-directions", type=Path, default=Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08/factor_directions.json"))
    parser.add_argument("--factor-dir", type=Path, default=Path("/cloud/factor_synthesis/factors/h5"))
    parser.add_argument("--market-h5", type=Path, default=Path("/cloud/hdf5/historical/china_astock_2018.h5"))
    parser.add_argument("--old-panel", type=Path, default=Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08_panel"))
    parser.add_argument("--old-execution", type=Path, default=Path("/cloud/E2EAI/data/execution_returns/execution_1d_return_verified_2018_2021.npy"))
    parser.add_argument("--old-benchmark", type=Path, default=Path("/cloud/E2EAI/data/benchmark/csi500_execution_1d_return.npy"))
    parser.add_argument("--output", type=Path, default=Path("/cloud/E2EAI/data/daily_strategy_fixed64_corr08_panel_2018_2026"))
    parser.add_argument("--block-dates", type=int, default=32)
    args = parser.parse_args()
    if args.block_dates < 1:
        parser.error("--block-dates must be positive")
    print(build(args))


if __name__ == "__main__":
    main()
