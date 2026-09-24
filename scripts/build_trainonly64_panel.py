"""Build a frozen train-only 64-factor daily-strategy panel without training.

Selection uses only the actual, purged training decision dates supplied by the
daily-strategy configuration.  Factor values are processed independently by
date, and their train-only directions are persisted outside the raw panel.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from e2eai.config import load_config
from e2eai.data.train_only_selection import select_train_only_factors


_WORKER_CONTEXT: dict[str, Any] = {}


def _decode(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S" or array.dtype == object:
        return np.asarray(
            [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in array],
            dtype=str,
        )
    return array.astype(str)


def _as_dates(values: np.ndarray) -> pd.DatetimeIndex:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        # Raw factor H5 stores its date axis as nanosecond Unix timestamps,
        # while the panel and pricing H5 use YYYYMMDD integers.  Preserve both
        # established encodings rather than interpreting nanoseconds as dates.
        if array.size and int(np.nanmax(array)) > 10**12:
            return pd.DatetimeIndex(pd.to_datetime(array, unit="ns"))
        return pd.DatetimeIndex(pd.to_datetime(array.astype(str), format="%Y%m%d"))
    return pd.DatetimeIndex(pd.to_datetime(_decode(array), errors="raise"))


def _factor_dataset(store: h5py.File) -> h5py.Dataset:
    if "data" in store and "block0_values" in store["data"]:
        return store["data/block0_values"]
    if "block0_values" in store:
        return store["block0_values"]
    raise KeyError("Factor H5 must contain data/block0_values or block0_values")


def _factor_axes(store: h5py.File) -> tuple[np.ndarray, pd.DatetimeIndex]:
    group = store["data"] if "data" in store else store
    return _decode(np.asarray(group["axis0"][:])), _as_dates(np.asarray(group["axis1"][:]))


def _read_aligned_factor(
    path: Path,
    decision_dates: pd.DatetimeIndex,
    panel_assets: np.ndarray,
    *,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Read one raw candidate in exact panel date/asset order, with NaN for gaps."""
    with h5py.File(path, "r") as store:
        factor_assets, factor_dates = _factor_axes(store)
        date_lookup = {date: index for index, date in enumerate(factor_dates)}
        asset_lookup = {asset: index for index, asset in enumerate(factor_assets)}
        date_rows = np.asarray([date_lookup.get(date, -1) for date in decision_dates], dtype=np.int64)
        asset_columns = np.asarray([asset_lookup.get(asset, -1) for asset in panel_assets], dtype=np.int64)
        output = np.full((len(decision_dates), len(panel_assets)), np.nan, dtype=dtype)
        if not np.any(date_rows >= 0) or not np.any(asset_columns >= 0):
            return output
        valid_rows = np.flatnonzero(date_rows >= 0)
        source_rows = date_rows[valid_rows]
        if np.any(np.diff(source_rows) < 0):
            raise ValueError(f"Factor dates are not monotonic for {path.name}")
        values = np.asarray(_factor_dataset(store)[source_rows, :], dtype=dtype)
        valid_columns = np.flatnonzero(asset_columns >= 0)
        output[np.ix_(valid_rows, valid_columns)] = values[:, asset_columns[valid_columns]]
    return output


def _init_worker(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _daily_rankic_worker(path_text: str) -> dict[str, Any]:
    """Compute one candidate's daily Spearman IC over train-only dates."""
    path = Path(path_text)
    context = _WORKER_CONTEXT
    try:
        raw = _read_aligned_factor(
            path,
            pd.DatetimeIndex(context["decision_dates"]),
            np.asarray(context["panel_assets"]),
            dtype=np.float64,
        )
        positions = np.asarray(context["asset_positions"], dtype=np.int64)
        safe_positions = np.maximum(positions, 0)
        x = np.take_along_axis(raw, safe_positions, axis=1)
        x = np.where(positions >= 0, x, np.nan)
        y = np.asarray(context["execution_values"], dtype=np.float64)
        base = np.asarray(context["base_universe_mask"], dtype=bool)
        valid = base & np.isfinite(x) & np.isfinite(y)
        ranked_x = rankdata(np.where(valid, x, np.nan), axis=1, method="average", nan_policy="omit")
        ranked_y = rankdata(np.where(valid, y, np.nan), axis=1, method="average", nan_policy="omit")
        count = valid.sum(axis=1).astype(np.float64)
        safe_count = np.maximum(count, 1.0)
        mean_x = np.where(valid, ranked_x, 0.0).sum(axis=1) / safe_count
        mean_y = np.where(valid, ranked_y, 0.0).sum(axis=1) / safe_count
        centered_x = np.where(valid, ranked_x - mean_x[:, None], 0.0)
        centered_y = np.where(valid, ranked_y - mean_y[:, None], 0.0)
        denominator = np.sqrt((centered_x * centered_x).sum(axis=1) * (centered_y * centered_y).sum(axis=1))
        ic = np.divide(
            (centered_x * centered_y).sum(axis=1),
            denominator,
            out=np.full(len(count), np.nan, dtype=np.float64),
            where=(count >= 3) & (denominator > 1e-12),
        )
        return {"factor_name": path.stem, "path": str(path), "daily_rankic": ic, "error": None}
    except Exception as exc:  # retain every candidate in the audit output
        return {
            "factor_name": path.stem,
            "path": str(path),
            "daily_rankic": np.full(len(context["decision_dates"]), np.nan, dtype=np.float64),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _actual_train_indices(config: Any, dates: pd.DatetimeIndex) -> tuple[np.ndarray, dict[str, Any]]:
    raw_train = np.flatnonzero(
        (dates >= pd.Timestamp(config.data.train_start))
        & (dates <= pd.Timestamp(config.data.train_end))
    )
    raw_validation = np.flatnonzero(
        (dates >= pd.Timestamp(config.data.validation_start))
        & (dates <= pd.Timestamp(config.data.validation_end))
    )
    if not len(raw_train) or not len(raw_validation):
        raise ValueError("Configured train/validation ranges are absent from the source panel")
    offsets = config.data.label_exit_offsets or config.data.panel_return_horizons
    max_offset = max(int(value) for value in offsets)
    actual = raw_train[raw_train + max_offset < raw_validation.min()]
    if not len(actual):
        raise ValueError("Purging removed every training decision date")
    return actual, {
        "requested_train_start": config.data.train_start,
        "requested_train_end": config.data.train_end,
        "actual_train_start": str(dates[actual[0]].date()),
        "actual_train_end": str(dates[actual[-1]].date()),
        "n_train_dates": int(len(actual)),
        "purge_rule": {
            "purge_label_overlap": bool(config.data.purge_label_overlap),
            "max_label_exit_offset": int(max_offset),
            "criterion": "decision_index + max_label_exit_offset < first_validation_index",
        },
    }


def _pricing_base_arrays(
    market_path: Path,
    dates: pd.DatetimeIndex,
    assets: np.ndarray,
    *,
    block_dates: int,
) -> dict[str, np.ndarray]:
    """Recreate current-date tradability without inheriting old factor coverage."""
    with h5py.File(market_path, "r") as market:
        market_dates = _as_dates(np.asarray(market["tradeDate"][:]))
        market_assets = _decode(np.asarray(market["ticker"][:]))
        date_lookup = {date: index for index, date in enumerate(market_dates)}
        asset_lookup = {asset: index for index, asset in enumerate(market_assets)}
        rows = np.asarray([date_lookup.get(date, -1) for date in dates], dtype=np.int64)
        columns = np.asarray([asset_lookup.get(asset, -1) for asset in assets], dtype=np.int64)
        if np.any(rows < 0) or np.any(columns < 0):
            raise ValueError("Pricing H5 cannot be aligned to the selected panel date/asset axes")
        shape = (len(dates), len(assets))
        base = np.zeros(shape, dtype=bool)
        market_cap = np.full(shape, np.nan, dtype=np.float32)
        free_share = np.full(shape, np.nan, dtype=np.float32)
        for start in range(0, len(dates), block_dates):
            stop = min(start + block_dates, len(dates))
            selected_rows = rows[start:stop]
            def read(name: str) -> np.ndarray:
                return np.asarray(market[name][selected_rows, :], dtype=np.float32)[:, columns]
            open_ = read("open")
            close = read("close")
            volume = read("volume")
            status = read("status")
            cap = read("marketCapital")
            free = read("freeTradableShare")
            base[start:stop] = (
                np.isfinite(status)
                & (status == 1)
                & np.isfinite(open_)
                & (open_ > 0)
                & np.isfinite(close)
                & (close > 0)
                & np.isfinite(volume)
                & (volume > 100)
                & np.isfinite(cap)
                & (cap > 0)
                & np.isfinite(free)
                & (free > 0)
            )
            market_cap[start:stop] = cap
            free_share[start:stop] = free
    return {"base_tradable": base, "market_cap": market_cap, "free_share": free_share}


def _selection_universe(
    dates: pd.DatetimeIndex,
    assets: np.ndarray,
    base_tradable: np.ndarray,
    execution_returns: np.ndarray,
    membership_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(membership_path, allow_pickle=False) as archive:
        membership_dates = _decode(archive["dates"])
        membership_assets = _decode(archive["asset_ids"])
        members = np.asarray(archive["members"], dtype=bool)
    date_lookup = {value: index for index, value in enumerate(membership_dates)}
    asset_lookup = {value: index for index, value in enumerate(membership_assets)}
    rows = np.asarray([date_lookup.get(str(date.date()), -1) for date in dates], dtype=np.int64)
    columns = np.asarray([asset_lookup.get(asset, -1) for asset in assets], dtype=np.int64)
    if np.any(rows < 0) or np.any(columns < 0):
        raise ValueError("Historical CSI500 membership cannot be aligned to selection axes")
    member_rows = members[rows][:, columns]
    counts = member_rows.sum(axis=1)
    max_count = int(counts.max())
    positions = np.full((len(dates), max_count), -1, dtype=np.int64)
    for row in range(len(dates)):
        positions[row, : counts[row]] = np.flatnonzero(member_rows[row])
    safe_positions = np.maximum(positions, 0)
    base = np.take_along_axis(base_tradable, safe_positions, axis=1)
    execution = np.take_along_axis(execution_returns, safe_positions, axis=1)
    active = (positions >= 0) & base & np.isfinite(execution)
    return positions, execution, active


def _copy_slice(source: Path, destination: Path, count: int, *, block_dates: int) -> None:
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    target = np.lib.format.open_memmap(
        destination, mode="w+", dtype=values.dtype, shape=(count, *values.shape[1:])
    )
    for start in range(0, count, block_dates):
        stop = min(start + block_dates, count)
        target[start:stop] = values[start:stop]
    target.flush()
    del target


def _build_panel(
    *,
    destination: Path,
    panel_dates: pd.DatetimeIndex,
    panel_assets: np.ndarray,
    selected_names: list[str],
    candidate_paths: dict[str, Path],
    pricing_arrays: dict[str, np.ndarray],
    execution_source: Path,
    forward_source: Path,
    benchmark_source: Path,
    source_panel: Path,
    block_dates: int,
    selection_metadata: dict[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    count = len(panel_dates)
    num_assets = len(panel_assets)
    _copy_slice(execution_source, destination / "execution_1d_return.npy", count, block_dates=block_dates)
    _copy_slice(forward_source, destination / forward_source.name, count, block_dates=block_dates)
    _copy_slice(benchmark_source, destination / benchmark_source.name, count, block_dates=block_dates)
    for name in ("forward_return_start_dates.npy", "forward_return_end_dates.npy"):
        _copy_slice(source_panel / name, destination / name, count, block_dates=block_dates)
    np.save(destination / "dates.npy", np.asarray(panel_dates.strftime("%Y-%m-%d"), dtype="U10"))
    np.save(destination / "asset_ids.npy", panel_assets.astype(str))
    np.save(destination / "alpha_names.npy", np.asarray(selected_names, dtype="U"))
    base = np.asarray(pricing_arrays["base_tradable"], dtype=bool)
    np.save(destination / "base_tradable_mask.npy", base)
    np.save(destination / "market_cap.npy", pricing_arrays["market_cap"].astype(np.float32))
    np.save(destination / "free_tradable_share.npy", pricing_arrays["free_share"].astype(np.float32))
    np.save(
        destination / "free_float_market_cap.npy",
        (pricing_arrays["market_cap"] * pricing_arrays["free_share"]).astype(np.float32),
    )
    alpha = np.lib.format.open_memmap(
        destination / "alpha_tensor.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, len(selected_names), num_assets),
    )
    factor_valid = np.lib.format.open_memmap(
        destination / "factor_valid_mask.npy",
        mode="w+",
        dtype=bool,
        shape=(count, len(selected_names), num_assets),
    )
    coverage = np.zeros((count, num_assets), dtype=np.uint8)
    for factor_index, name in enumerate(selected_names):
        # Keep H5 values in float64 through the within-date transformations.
        # The final normalised panel is explicitly written as float32 below.
        raw = _read_aligned_factor(
            candidate_paths[name], panel_dates, panel_assets, dtype=np.float64
        )
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
        print(f"panel_factor={factor_index + 1}/{len(selected_names)} {name}", flush=True)
    alpha.flush()
    factor_valid.flush()
    del alpha, factor_valid
    execution = np.load(destination / "execution_1d_return.npy", mmap_mode="r", allow_pickle=False)
    required = int(np.ceil(0.90 * len(selected_names)))
    eligible = base & (coverage >= required) & np.isfinite(execution)
    np.save(destination / "eligible_mask.npy", eligible)
    np.save(destination / "normal_eligible_mask.npy", eligible)
    metadata = {
        "schema": "historical_train_only_panel_v1",
        "selection_period": selection_metadata["selection_period"],
        "actual_selection_start": selection_metadata["actual_selection_start"],
        "actual_selection_end": selection_metadata["actual_selection_end"],
        "selection_target": "execution_1d_return = adjusted VWAP[t+2] / adjusted VWAP[t+1] - 1",
        "selection_metric": "abs(train_mean_daily_cross_sectional_Spearman_RankIC)",
        "top_k": len(selected_names),
        "direction_source": selection_metadata["factor_directions_path"],
        "direction_definition": "sign(train_mean_rankic), with exact zero assigned +1; directions are external and NOT applied to alpha_tensor",
        "candidate_factor_count": selection_metadata["candidate_factor_count"],
        "selected_factor_count": len(selected_names),
        "lookahead_safe": True,
        "raw_processed_factor_values": True,
        "same_day_preprocessing": "per-date median fill, 1/99 winsorisation, cross-sectional z-score",
        "min_feature_coverage": 0.90,
        "source_panel": str(source_panel),
        "execution_source": str(execution_source),
        "forward_return_source": str(forward_source),
        "benchmark_source": str(benchmark_source),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_audit(path: Path, selection: dict[str, Any], panel_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Train-only 64-factor information-timing audit — PASS_WITH_UPSTREAM_WARNING\n\n"
        "## Confirmed protections\n\n"
        f"- Selection dates are frozen to {selection['actual_train_start']} through "
        f"{selection['actual_train_end']} ({selection['n_train_dates']} decision dates), "
        "using the identical longest-label purge rule as model training.\n"
        "- Candidate RankIC, Top-64 selection, and direction use only execution_1d_return on those dates. "
        "No 2021 validation, 2022 test, or later label row is supplied to the selection function.\n"
        "- Ties are resolved by descending absolute train mean RankIC then ascending factor name.\n"
        "- The panel stores raw processed factors without multiplying by direction. Directions are external, frozen artifacts.\n"
        "- Factor H5 values are aligned by the same decision date t. Fill, winsorisation and normalisation are all performed cross-sectionally within t.\n\n"
        "## Upstream warning\n\n"
        "The raw factor H5 formula implementations, their rolling-window conventions, and financial-data publication-time rules are not fully available in this project. "
        "No explicit future row is used by this panel builder, but that upstream provenance still requires a separate source audit.\n\n"
        "## Result\n\n"
        "**PASS_WITH_UPSTREAM_WARNING** for the repaired selection/direction P0.\n\n"
        f"Panel: `{panel_path}`\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/daily_strategy.yaml"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("/cloud/factor_synthesis/factors/h5"))
    parser.add_argument("--pricing-h5", type=Path, default=Path("/cloud/hdf5/historical/china_astock_2018.h5"))
    parser.add_argument("--selection-output", type=Path, default=Path("/cloud/E2EAI/data/historical_train_only_selection"))
    parser.add_argument("--panel-output", type=Path, default=Path("/cloud/E2EAI/data/historical_train_only_panel"))
    parser.add_argument("--audit-output", type=Path, default=Path("output/daily_strategy_diagnostics/factor_information_timing_audit_trainonly64.md"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--block-dates", type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1 or args.block_dates < 1:
        parser.error("workers and block-dates must be positive")
    if args.selection_output.exists() or args.panel_output.exists():
        raise FileExistsError("Refusing to overwrite an existing train-only selection or panel output")

    config = load_config(args.config)
    source_panel = Path(config.data.path or "")
    source_dates = _as_dates(np.load(source_panel / "dates.npy", allow_pickle=False))
    source_assets = _decode(np.load(source_panel / "asset_ids.npy", allow_pickle=False))
    train_indices, selection_dates = _actual_train_indices(config, source_dates)
    selection_dates_path = args.selection_output / "factor_selection_train_dates.json"
    args.selection_output.mkdir(parents=True, exist_ok=False)
    selection_dates_path.write_text(json.dumps(selection_dates, ensure_ascii=False, indent=2), encoding="utf-8")

    execution_source = Path(config.data.execution_return_path or "")
    execution = np.load(execution_source, mmap_mode="r", allow_pickle=False)
    train_execution = np.asarray(execution[train_indices], dtype=np.float64)
    pricing_train = _pricing_base_arrays(
        args.pricing_h5, source_dates[train_indices], source_assets, block_dates=args.block_dates
    )
    asset_positions, selected_execution, base_mask = _selection_universe(
        source_dates[train_indices],
        source_assets,
        pricing_train["base_tradable"],
        train_execution,
        Path(config.data.membership_path or ""),
    )
    candidate_paths = {path.stem: path for path in sorted(args.candidate_dir.glob("*.h5"))}
    if len(candidate_paths) < 64:
        raise ValueError("The raw candidate factor pool has fewer than 64 H5 files")
    worker_context = {
        "decision_dates": source_dates[train_indices].to_numpy(),
        "panel_assets": source_assets,
        "asset_positions": asset_positions,
        "execution_values": selected_execution,
        "base_universe_mask": base_mask,
    }
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(worker_context,)
    ) as executor:
        futures = {executor.submit(_daily_rankic_worker, str(path)): name for name, path in candidate_paths.items()}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed % 25 == 0 or completed == len(futures):
                print(f"selection_candidates={completed}/{len(futures)}", flush=True)
    results.sort(key=lambda row: row["factor_name"])
    daily_rankic = np.column_stack([row["daily_rankic"] for row in results])
    statistics = select_train_only_factors(
        [row["factor_name"] for row in results],
        daily_rankic,
        np.ones(daily_rankic.shape[0], dtype=bool),
        top_k=64,
    )
    source_info = pd.DataFrame(
        {
            "factor_name": [row["factor_name"] for row in results],
            "candidate_factor_source_path": [row["path"] for row in results],
            "read_error": [row["error"] for row in results],
        }
    )
    statistics = statistics.merge(source_info, on="factor_name", how="left", validate="one_to_one")
    selection_path = args.selection_output / "factor_selection.csv"
    statistics.to_csv(selection_path, index=False, encoding="utf-8-sig")
    selected = statistics.loc[statistics["selected"]].sort_values("rank")
    if len(selected) != 64 or selected["train_mean_rankic"].isna().any():
        raise RuntimeError("Train-only ranking did not produce 64 finite selected factors")
    selected_names = selected["factor_name"].astype(str).tolist()
    directions = {
        name: int(direction)
        for name, direction in zip(selected_names, selected["train_only_direction"].astype(int))
    }
    direction_path = args.selection_output / "factor_directions.json"
    direction_path.write_text(json.dumps(directions, ensure_ascii=False, indent=2), encoding="utf-8")
    factor_list_path = args.selection_output / "factor_list.json"
    factor_list_path.write_text(
        json.dumps({"factor_names": selected_names}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output_indices = np.flatnonzero(source_dates <= pd.Timestamp(config.data.validation_end))
    pricing_output = _pricing_base_arrays(
        args.pricing_h5, source_dates[output_indices], source_assets, block_dates=args.block_dates
    )
    build_metadata = {
        "selection_period": "actual purged train decision dates",
        "actual_selection_start": selection_dates["actual_train_start"],
        "actual_selection_end": selection_dates["actual_train_end"],
        "n_train_dates": selection_dates["n_train_dates"],
        "candidate_factor_count": len(candidate_paths),
        "factor_directions_path": str(direction_path),
    }
    _build_panel(
        destination=args.panel_output,
        panel_dates=source_dates[output_indices],
        panel_assets=source_assets,
        selected_names=selected_names,
        candidate_paths=candidate_paths,
        pricing_arrays=pricing_output,
        execution_source=execution_source,
        forward_source=Path(config.data.return_path or ""),
        benchmark_source=Path(config.data.benchmark_path or ""),
        source_panel=source_panel,
        block_dates=args.block_dates,
        selection_metadata=build_metadata,
    )
    old_names = _decode(np.load(source_panel / "alpha_names.npy", allow_pickle=False))
    old_directions = np.load(source_panel / "alpha_directions.npy", allow_pickle=False)
    old_rank = {name: index + 1 for index, name in enumerate(old_names)}
    old_direction = {name: int(value) for name, value in zip(old_names, old_directions)}
    new_rank = {name: int(rank) for name, rank in zip(selected_names, selected["rank"])}
    comparison_names = sorted(set(old_names.tolist()) | set(selected_names))
    comparison = pd.DataFrame(
        {
            "factor_name": comparison_names,
            "in_old_top64": [name in old_rank for name in comparison_names],
            "in_new_top64": [name in new_rank for name in comparison_names],
            "old_rank_if_known": [old_rank.get(name) for name in comparison_names],
            "new_rank": [new_rank.get(name) for name in comparison_names],
            "old_direction_if_known": [old_direction.get(name) for name in comparison_names],
            "new_direction": [directions.get(name) for name in comparison_names],
        }
    )
    comparison["direction_changed"] = (
        comparison["in_old_top64"]
        & comparison["in_new_top64"]
        & (comparison["old_direction_if_known"] != comparison["new_direction"])
    )
    comparison_path = args.selection_output / "factor_selection_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    summary = {
        "overlap_count": int((comparison["in_old_top64"] & comparison["in_new_top64"]).sum()),
        "added_count": int((~comparison["in_old_top64"] & comparison["in_new_top64"]).sum()),
        "removed_count": int((comparison["in_old_top64"] & ~comparison["in_new_top64"]).sum()),
        "direction_flip_count": int(comparison["direction_changed"].sum()),
    }
    (args.selection_output / "factor_selection_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "selection_period": build_metadata["selection_period"],
        "actual_selection_start": build_metadata["actual_selection_start"],
        "actual_selection_end": build_metadata["actual_selection_end"],
        "selection_target": "execution_1d_return = adjusted VWAP[t+2] / adjusted VWAP[t+1] - 1",
        "selection_metric": "abs(train_mean_daily_cross_sectional_Spearman_RankIC)",
        "candidate_count": len(candidate_paths),
        "selected_count": 64,
        "factor_list_path": str(factor_list_path),
        "factor_directions_path": str(direction_path),
        "factor_selection_path": str(selection_path),
        "factor_names": selected_names,
        "file_hashes": {
            "factor_list.json": _sha256(factor_list_path),
            "factor_directions.json": _sha256(direction_path),
            "factor_selection.csv": _sha256(selection_path),
        },
        "creation_time_utc": datetime.now(UTC).isoformat(),
        "code_version_git_commit": _git_commit(Path.cwd()),
        "lookahead_safe": True,
        "upstream_warning": "Raw factor H5 formula/publication-time provenance requires a separate audit.",
    }
    manifest_path = args.selection_output / "factor_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_audit(args.audit_output, selection_dates, args.panel_output)
    print(json.dumps({"selection_output": str(args.selection_output), "panel_output": str(args.panel_output), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
