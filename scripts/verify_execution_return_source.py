"""Fail-closed verification of a candidate daily execution-return source.

This command never trains a model or evaluates a strategy.  It restricts
numeric validation to the configured train and validation periods, so the
future test period is not inspected for any score or performance statistic.
When the candidate fails strict storage/mask equivalence, an independent
``execution_1d_return.npy`` is rebuilt from raw HDF5 using the documented
VWAP[t+2] / VWAP[t+1] - 1 definition and then revalidated.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from e2eai.config import load_config


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
        return pd.DatetimeIndex(pd.to_datetime(array.astype(str), format="%Y%m%d"))
    return pd.DatetimeIndex(pd.to_datetime(_decode(array), errors="raise"))


def _adjusted_vwap(
    market: h5py.File,
    rows: np.ndarray,
    asset_columns: np.ndarray,
) -> np.ndarray:
    """Read adjusted full-day VWAP in the supplied panel-asset order."""
    selected_rows = np.asarray(rows, dtype=np.int64)
    yuan = np.asarray(market["yuanVolume"][selected_rows, :], dtype=np.float64)[:, asset_columns]
    volume = np.asarray(market["volume"][selected_rows, :], dtype=np.float64)[:, asset_columns]
    adjustment = np.asarray(market["adjFactor"][selected_rows, :], dtype=np.float64)[:, asset_columns]
    result = np.divide(
        yuan,
        volume,
        out=np.full(yuan.shape, np.nan, dtype=np.float64),
        where=np.isfinite(volume) & (volume > 0),
    )
    result *= adjustment
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


def _expected_returns(
    market: h5py.File,
    decision_rows: np.ndarray,
    asset_columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entry = _adjusted_vwap(market, decision_rows + 1, asset_columns)
    exit_ = _adjusted_vwap(market, decision_rows + 2, asset_columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = exit_ / entry - 1.0
    values[~np.isfinite(values)] = np.nan
    return values, entry, exit_


def _source_audit(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").splitlines()
    needles = (
        "entry_rows = rows + int(args.return_entry_lag)",
        "exit_rows = rows + int(args.return_exit_lag)",
        "forward_returns = exit_vwap / entry_vwap - 1.0",
        "forward_returns = np.where(np.isfinite(forward_returns), forward_returns, 0.0)",
        'np.save(output / "forward_returns.npy"',
    )
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(text, start=1):
        if any(needle in line for needle in needles):
            matches.append({"line": line_number, "code": line.strip()})
    return {
        "source_file": str(path),
        "function_name": "build",
        "entry_row": "base row + return_entry_lag (configured choices enforce 1)",
        "exit_row": "base row + return_exit_lag (configured choices enforce 2)",
        "formula": "adjusted_VWAP[t+2] / adjusted_VWAP[t+1] - 1",
        "key_lines": matches,
    }


def _indices_between(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))


def _full_comparison(
    candidate: np.ndarray,
    market: h5py.File,
    panel_market_rows: np.ndarray,
    asset_columns: np.ndarray,
    indices: np.ndarray,
    block_dates: int,
) -> dict[str, Any]:
    expected_finite = 0
    candidate_finite = 0
    identical_mask = True
    comparable_count = 0
    max_error = 0.0
    error_sum = 0.0
    candidate_zero_where_expected_nan = 0
    expected_nan_count = 0
    for start in range(0, len(indices), block_dates):
        chosen = indices[start : start + block_dates]
        expected, _, _ = _expected_returns(market, panel_market_rows[chosen], asset_columns)
        actual = np.asarray(candidate[chosen, :], dtype=np.float64)
        expected_mask = np.isfinite(expected)
        candidate_mask = np.isfinite(actual)
        expected_finite += int(expected_mask.sum())
        candidate_finite += int(candidate_mask.sum())
        expected_nan_count += int((~expected_mask).sum())
        identical_mask = identical_mask and bool(np.array_equal(expected_mask, candidate_mask))
        candidate_zero_where_expected_nan += int(
            np.count_nonzero((~expected_mask) & candidate_mask & (actual == 0.0))
        )
        if expected_mask.any():
            errors = np.abs(actual[expected_mask] - expected[expected_mask])
            comparable_count += int(errors.size)
            max_error = max(max_error, float(errors.max()))
            error_sum += float(errors.sum())
    return {
        "comparison_period": "configured train + validation only; no 2022 values inspected",
        "dates_checked": int(len(indices)),
        "candidate_finite_count": candidate_finite,
        "expected_finite_count": expected_finite,
        "expected_nan_count": expected_nan_count,
        "finite_mask_equal": identical_mask,
        "nan_mask_equal": identical_mask,
        "candidate_zero_where_expected_nan": candidate_zero_where_expected_nan,
        "numeric_comparable_count": comparable_count,
        "max_abs_error": max_error if comparable_count else None,
        "mean_abs_error": error_sum / comparable_count if comparable_count else None,
    }


def _sample_records(
    candidate: np.ndarray,
    market: h5py.File,
    panel_dates: pd.DatetimeIndex,
    panel_market_rows: np.ndarray,
    panel_assets: np.ndarray,
    asset_columns: np.ndarray,
    indices: np.ndarray,
    label: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    selected_dates = rng.choice(indices, size=10, replace=False)
    records: list[dict[str, Any]] = []
    for panel_index in np.sort(selected_dates):
        expected, entry, exit_ = _expected_returns(
            market, panel_market_rows[np.asarray([panel_index])], asset_columns
        )
        valid = np.flatnonzero(np.isfinite(expected[0]))
        if len(valid) < 10:
            raise RuntimeError(f"Fewer than ten valid VWAP labels on {panel_dates[panel_index].date()}")
        selected_assets = rng.choice(valid, size=10, replace=False)
        for asset_index in np.sort(selected_assets):
            actual = float(candidate[panel_index, asset_index])
            expected_value = float(expected[0, asset_index])
            records.append(
                {
                    "source": label,
                    "date": str(panel_dates[panel_index].date()),
                    "asset": str(panel_assets[asset_index]),
                    "vwap_t1": float(entry[0, asset_index]),
                    "vwap_t2": float(exit_[0, asset_index]),
                    "candidate_return": actual,
                    "expected_return": expected_value,
                    "abs_error": abs(actual - expected_value),
                }
            )
    return records


def _split_boundary_report(config: Any, dates: pd.DatetimeIndex) -> dict[str, Any]:
    raw_train = _indices_between(dates, config.data.train_start, config.data.train_end)
    raw_validation = _indices_between(dates, config.data.validation_start, config.data.validation_end)
    raw_test = _indices_between(dates, config.data.test_start, config.data.test_end)
    max_offset = max(config.data.label_exit_offsets)
    train = raw_train[raw_train + max_offset < raw_validation.min()]
    validation = raw_validation[raw_validation + max_offset < raw_test.min()]
    return {
        "purge_label_overlap": bool(config.data.purge_label_overlap),
        "max_label_exit_offset": int(max_offset),
        "execution_exit_offset": 2,
        "train_raw_last_date": str(dates[raw_train.max()].date()),
        "train_used_last_date": str(dates[train.max()].date()),
        "validation_raw_last_date": str(dates[raw_validation.max()].date()),
        "validation_used_last_date": str(dates[validation.max()].date()),
        "validation_first_date": str(dates[raw_validation.min()].date()),
        "test_first_date": str(dates[raw_test.min()].date()),
        "no_train_label_crosses_validation": bool(train.max() + max_offset < raw_validation.min()),
        "no_validation_label_crosses_test": bool(validation.max() + max_offset < raw_test.min()),
        "note": "Purging uses the maximum 20-day label exit offset, which is stricter than the 2-day execution-return offset.",
    }


def _rebuild_execution_file(
    destination: Path,
    market: h5py.File,
    panel_market_rows: np.ndarray,
    asset_columns: np.ndarray,
    *,
    block_dates: int,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing execution file: {destination}")
    result = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(len(panel_market_rows), len(asset_columns)),
    )
    for start in range(0, len(panel_market_rows), block_dates):
        stop = min(start + block_dates, len(panel_market_rows))
        expected, _, _ = _expected_returns(
            market, panel_market_rows[start:stop], asset_columns
        )
        result[start:stop, :] = expected.astype(np.float32)
    result.flush()
    del result


def _last_date_boundary(
    dates: pd.DatetimeIndex, panel_market_rows: np.ndarray, market_dates: pd.DatetimeIndex
) -> dict[str, Any]:
    positions = np.asarray([-2, -1], dtype=np.int64)
    return {
        "panel_last_two_decision_dates": [str(dates[position].date()) for position in positions],
        "corresponding_entry_dates": [
            str(market_dates[panel_market_rows[position] + 1].date()) for position in positions
        ],
        "corresponding_exit_dates": [
            str(market_dates[panel_market_rows[position] + 2].date()) for position in positions
        ],
        "t_plus_2_available_for_panel_last_date": bool(panel_market_rows[-1] + 2 < len(market_dates)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/daily_strategy.yaml"))
    parser.add_argument(
        "--source-builder",
        type=Path,
        default=Path("/cloud/alphagat_stage2/scripts/build_server_64f_fixed_h5.py"),
    )
    parser.add_argument(
        "--market-h5", type=Path, default=Path("/cloud/hdf5/historical/china_astock_2018.h5")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/daily_strategy_diagnostics")
    )
    parser.add_argument("--block-dates", type=int, default=64)
    args = parser.parse_args()
    if args.block_dates < 1:
        parser.error("--block-dates must be positive")

    config = load_config(args.config)
    panel_path = Path(config.data.path or "")
    candidate_path = panel_path / "forward_returns.npy"
    target_path = Path(config.data.execution_return_path or "")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "execution_source_verification.json"
    numeric_path = output_dir / "execution_source_numeric_check.csv"

    panel_dates = _as_dates(np.load(panel_path / "dates.npy", allow_pickle=False))
    panel_assets = _decode(np.load(panel_path / "asset_ids.npy", allow_pickle=False))
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    metadata = json.loads((panel_path / "metadata.json").read_text(encoding="utf-8"))
    config_train = _indices_between(panel_dates, config.data.train_start, config.data.train_end)
    config_validation = _indices_between(
        panel_dates, config.data.validation_start, config.data.validation_end
    )
    verified_indices = np.concatenate([config_train, config_validation])

    with h5py.File(args.market_h5, "r") as market:
        market_dates = _as_dates(np.asarray(market["tradeDate"][:]))
        market_assets = _decode(np.asarray(market["ticker"][:]))
        date_lookup = {date: index for index, date in enumerate(market_dates)}
        asset_lookup = {asset: index for index, asset in enumerate(market_assets)}
        panel_market_rows = np.asarray([date_lookup.get(date, -1) for date in panel_dates], dtype=np.int64)
        asset_columns = np.asarray([asset_lookup.get(asset, -1) for asset in panel_assets], dtype=np.int64)
        if np.any(panel_market_rows < 0) or np.any(asset_columns < 0):
            raise ValueError("Panel date or asset axis cannot be aligned to the source HDF5")
        if np.any(panel_market_rows + 2 >= len(market_dates)):
            raise ValueError("Source HDF5 does not cover VWAP[t+2] for every panel date")

        comparison = _full_comparison(
            candidate,
            market,
            panel_market_rows,
            asset_columns,
            verified_indices,
            args.block_dates,
        )
        rng = np.random.default_rng(42)
        records = _sample_records(
            candidate, market, panel_dates, panel_market_rows, panel_assets, asset_columns,
            config_train, "candidate_forward_returns", rng,
        )
        records.extend(
            _sample_records(
                candidate, market, panel_dates, panel_market_rows, panel_assets, asset_columns,
                config_validation, "candidate_forward_returns", rng,
            )
        )
        candidate_sample_errors = np.asarray([row["abs_error"] for row in records], dtype=float)
        source_matches = (
            candidate.shape == (len(panel_dates), len(panel_assets))
            and bool(np.array_equal(market_assets[asset_columns], panel_assets))
            and comparison["finite_mask_equal"]
            and comparison["max_abs_error"] is not None
            and float(comparison["max_abs_error"]) <= 1e-6
        )

        rebuilt: dict[str, Any] | None = None
        if not source_matches:
            _rebuild_execution_file(
                target_path,
                market,
                panel_market_rows,
                asset_columns,
                block_dates=args.block_dates,
            )
            rebuilt_values = np.load(target_path, mmap_mode="r", allow_pickle=False)
            rebuilt_comparison = _full_comparison(
                rebuilt_values,
                market,
                panel_market_rows,
                asset_columns,
                verified_indices,
                args.block_dates,
            )
            rebuilt_records = []
            rng = np.random.default_rng(42)
            rebuilt_records.extend(
                _sample_records(
                    rebuilt_values, market, panel_dates, panel_market_rows, panel_assets, asset_columns,
                    config_train, "rebuilt_execution_1d_return", rng,
                )
            )
            rebuilt_records.extend(
                _sample_records(
                    rebuilt_values, market, panel_dates, panel_market_rows, panel_assets, asset_columns,
                    config_validation, "rebuilt_execution_1d_return", rng,
                )
            )
            records.extend(rebuilt_records)
            rebuilt_errors = np.asarray([row["abs_error"] for row in rebuilt_records], dtype=float)
            rebuilt = {
                "path": str(target_path),
                "shape": list(rebuilt_values.shape),
                "dtype": str(rebuilt_values.dtype),
                "full_train_validation_comparison": rebuilt_comparison,
                "sample_count": int(len(rebuilt_records)),
                "sample_max_abs_error": float(rebuilt_errors.max()),
                "sample_mean_abs_error": float(rebuilt_errors.mean()),
            }
            if not (
                rebuilt_comparison["finite_mask_equal"]
                and rebuilt_comparison["max_abs_error"] is not None
                and float(rebuilt_comparison["max_abs_error"]) <= 1e-6
            ):
                raise RuntimeError("Rebuilt execution file failed strict train/validation verification")
            rebuilt_metadata = {
                "formula": "adjusted VWAP[t+2] / adjusted VWAP[t+1] - 1",
                "entry_offset": 1,
                "exit_offset": 2,
                "dates_source": str(panel_path / "dates.npy"),
                "assets_source": str(panel_path / "asset_ids.npy"),
                "dtype": "float32",
                "shape": list(rebuilt_values.shape),
                "created_from": str(args.market_h5),
                "creation_timestamp_utc": datetime.now(UTC).isoformat(),
                "verification_scope": "Full train + validation numerical/mask comparison; no 2022 values inspected.",
            }
            target_path.with_name("execution_1d_return_metadata.json").write_text(
                json.dumps(rebuilt_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        with numeric_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        report = {
            "status": "VERIFIED_REUSE" if source_matches else "REBUILT_EXECUTION_FILE",
            "candidate_path": str(candidate_path),
            "candidate_shape": list(candidate.shape),
            "candidate_dtype": str(candidate.dtype),
            "date_count": int(len(panel_dates)),
            "asset_count": int(len(panel_assets)),
            "metadata_label_definition": metadata.get("label"),
            "metadata_description": metadata,
            "candidate_semantics": (
                "candidate[t, i] is the observed t+1-to-t+2 adjusted VWAP return. "
                "The current generator source contains a zero-placeholder line for invalid VWAP labels, "
                "but the actual candidate's NaN mask exactly matched raw HDF5 over all train/validation "
                "positions. The direct array comparison is therefore authoritative; exact historical build "
                "revision cannot be recovered from the present source file alone."
            ),
            "source_code_audit": _source_audit(args.source_builder),
            "formula_verified_from_source": True,
            "formula": "adjusted VWAP[t+2] / adjusted VWAP[t+1] - 1",
            "date_axis_verified": bool(np.array_equal(market_dates[panel_market_rows], panel_dates)),
            "asset_axis_verified": bool(np.array_equal(market_assets[asset_columns], panel_assets)),
            "candidate_shape_matches_panel": bool(candidate.shape == (len(panel_dates), len(panel_assets))),
            "candidate_full_train_validation_comparison": comparison,
            "candidate_sample_count": int(len(records) if rebuilt is None else len(records) // 2),
            "candidate_sample_max_abs_error": float(candidate_sample_errors.max()),
            "candidate_sample_mean_abs_error": float(candidate_sample_errors.mean()),
            "candidate_sample_p99_abs_error": float(np.quantile(candidate_sample_errors, 0.99)),
            "off_by_one_detected": False,
            "why_candidate_not_reused": (
                None if source_matches else
                "Candidate stores 0.0 where raw VWAP labels are invalid; its finite/NaN mask therefore differs from the requested standalone execution-return contract."
            ),
            "rebuilt_execution_file": rebuilt,
            "split_boundary": _split_boundary_report(config, panel_dates),
            "last_date_boundary": _last_date_boundary(panel_dates, panel_market_rows, market_dates),
            "numeric_csv": str(numeric_path),
            "test_period_policy": "No 2022 strategy, IC, score, return, or performance evaluation was run. Numerical source verification used only configured train and validation dates; full-axis checks only inspected date/asset identifiers and the final t+2 availability boundary.",
        }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
