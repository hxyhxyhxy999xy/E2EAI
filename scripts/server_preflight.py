"""Write a fail-closed preflight report for a real daily-strategy server run.

The script deliberately performs no fallback path substitution.  If any data
file configured as a core daily-strategy input is absent, it records the issue
and exits before loading data or starting model work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from e2eai.config import load_config


def _array_metadata(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return item
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        item.update(shape=list(values.shape), dtype=str(values.dtype))
    except Exception as exc:  # report a malformed core input as a failed preflight
        item["read_error"] = f"{type(exc).__name__}: {exc}"
    return item


def _panel_metadata(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.is_dir()}
    required = (
        "dates.npy",
        "asset_ids.npy",
        "alpha_tensor.npy",
        "alpha_names.npy",
        "factor_valid_mask.npy",
        "eligible_mask.npy",
    )
    item["arrays"] = {
        name.removesuffix(".npy"): _array_metadata(path / name) for name in required
    }
    if path.is_dir() and (path / "metadata.json").is_file():
        try:
            item["metadata"] = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        except Exception as exc:
            item["metadata_read_error"] = f"{type(exc).__name__}: {exc}"
    return item


def _json_metadata(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return item
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        item["keys"] = sorted(content) if isinstance(content, dict) else None
        item["content"] = content
    except Exception as exc:
        item["read_error"] = f"{type(exc).__name__}: {exc}"
    return item


def _factor_manifest_alignment(panel_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Fail closed unless the panel's ordered factor axis equals its manifest."""
    item: dict[str, Any] = {
        "path": str(panel_path / "alpha_names.npy"),
        "manifest_path": str(manifest_path),
        "exists": False,
    }
    if not (panel_path / "alpha_names.npy").is_file() or not manifest_path.is_file():
        return item
    try:
        names = np.asarray(np.load(panel_path / "alpha_names.npy", allow_pickle=False)).astype(str).tolist()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = [str(value) for value in manifest["factor_names"]]
        item.update(
            exists=(names == expected and len(names) == 64),
            panel_factor_count=len(names),
            manifest_factor_count=len(expected),
            ordered_names_match=(names == expected),
        )
    except Exception as exc:
        item["read_error"] = f"{type(exc).__name__}: {exc}"
    return item


def _external_array_alignment(panel_path: Path, config: Any) -> dict[str, Any]:
    """Check every externally configured return axis before any model work."""
    item: dict[str, Any] = {"exists": False}
    try:
        alpha = np.load(panel_path / "alpha_tensor.npy", mmap_mode="r", allow_pickle=False)
        dates = np.load(panel_path / "dates.npy", mmap_mode="r", allow_pickle=False)
        if alpha.ndim != 3 or alpha.shape[0] != len(dates):
            raise ValueError("Panel alpha_tensor and date axis are inconsistent")
        periods, _, assets = alpha.shape
        forward = np.load(Path(config.data.return_path), mmap_mode="r", allow_pickle=False)
        execution = np.load(
            Path(config.data.execution_return_path), mmap_mode="r", allow_pickle=False
        )
        benchmark = np.load(Path(config.data.benchmark_path), mmap_mode="r", allow_pickle=False)
        execution_only = (
            config.model.execution_horizon is not None
            and not config.data.panel_return_horizons
            and list(config.model.horizons) == [int(config.model.execution_horizon)]
        )
        expected_forward = (periods, assets) if execution_only else (
            periods, len(config.data.panel_return_horizons), assets
        )
        expected_execution = (periods, assets)
        allowed_benchmark = {(periods,)} if execution_only else {
            (periods,), (periods, len(config.data.panel_return_horizons))
        }
        item.update(
            exists=(
                forward.shape == expected_forward
                and execution.shape == expected_execution
                and benchmark.shape in allowed_benchmark
            ),
            expected={
                "forward_returns": list(expected_forward),
                "execution_1d_return": list(expected_execution),
                "benchmark_allowed": [list(shape) for shape in sorted(allowed_benchmark)],
            },
            observed={
                "forward_returns": list(forward.shape),
                "execution_1d_return": list(execution.shape),
                "benchmark": list(benchmark.shape),
            },
        )
    except Exception as exc:
        item["read_error"] = f"{type(exc).__name__}: {exc}"
    return item


def _date_range(path: Path, key: str | None = None) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        if key is None:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as archive:
                values = archive[key]
        if len(values) == 0:
            return None
        return {"start": str(values[0]), "end": str(values[-1])}
    except Exception:
        return None


def _industry_metadata(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return item
    try:
        with np.load(path, allow_pickle=False) as archive:
            item["keys"] = list(archive.files)
            for key in ("index", "columns", "values"):
                if key in archive:
                    item[key] = {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
    except Exception as exc:
        item["read_error"] = f"{type(exc).__name__}: {exc}"
    return item


def _finite_summary(values: np.ndarray, *, percentiles: bool = False) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    result: dict[str, Any] = {"finite_count": int(finite.size)}
    if finite.size == 0:
        result.update(mean=None, std=None)
        if percentiles:
            result.update(p01=None, p50=None, p99=None)
        return result
    result.update(mean=float(finite.mean()), std=float(finite.std(ddof=0)))
    if percentiles:
        p01, p50, p99 = np.quantile(finite, [0.01, 0.50, 0.99])
        result.update(p01=float(p01), p50=float(p50), p99=float(p99))
    return result


def _full_statistics(config: Any, panel_path: Path) -> dict[str, Any]:
    """Compute preflight diagnostics on train+validation only, never 2022."""
    dates = np.asarray(np.load(panel_path / "dates.npy", mmap_mode="r", allow_pickle=False))
    date_index = np.asarray([str(value) for value in dates])
    valid_dates = np.asarray(
        [
            index
            for index, value in enumerate(date_index)
            if str(config.data.train_start) <= value <= str(config.data.validation_end)
        ],
        dtype=np.int64,
    )
    if valid_dates.size == 0:
        raise ValueError("No train/validation dates available for preflight statistics")
    alpha = np.load(panel_path / "alpha_tensor.npy", mmap_mode="r", allow_pickle=False)
    names_path = panel_path / "alpha_names.npy"
    names = (
        np.asarray(np.load(names_path, allow_pickle=False)).astype(str)
        if names_path.is_file()
        else np.asarray([f"factor_{index:03d}" for index in range(alpha.shape[1])])
    )
    factor_stats: list[dict[str, Any]] = []
    for factor, name in enumerate(names):
        values = np.asarray(alpha[valid_dates, factor, :], dtype=np.float32)
        total = int(values.size)
        finite = np.isfinite(values)
        factor_stats.append(
            {
                "factor_name": str(name),
                "finite_ratio": float(finite.mean()),
                "nan_ratio": float(np.isnan(values).mean()),
                "zero_ratio": float(np.count_nonzero(values == 0.0) / total),
            }
        )

    execution = np.load(
        Path(config.data.execution_return_path), mmap_mode="r", allow_pickle=False
    )
    execution_values = np.asarray(execution[valid_dates, :], dtype=np.float64)
    execution_finite = np.isfinite(execution_values)
    execution_stats = _finite_summary(execution_values, percentiles=True)
    execution_stats["finite_ratio"] = float(execution_finite.mean())

    forward = np.load(Path(config.data.return_path), mmap_mode="r", allow_pickle=False)
    forward_stats: dict[str, Any] = {}
    for position, horizon in enumerate(config.data.panel_return_horizons):
        values = np.asarray(forward[valid_dates, position, :], dtype=np.float64)
        summary = _finite_summary(values)
        summary["finite_ratio"] = float(np.isfinite(values).mean())
        forward_stats[str(horizon)] = summary

    membership_path = Path(config.data.membership_path)
    with np.load(membership_path, allow_pickle=False) as membership:
        membership_dates = np.asarray(membership["dates"]).astype(str)
        membership_assets = np.asarray(membership["asset_ids"]).astype(str)
        membership_values = np.asarray(membership["members"], dtype=bool)
    panel_assets = np.asarray(np.load(panel_path / "asset_ids.npy", allow_pickle=False)).astype(str)
    date_positions = {value: index for index, value in enumerate(membership_dates)}
    asset_positions = {value: index for index, value in enumerate(membership_assets)}
    member_rows = np.asarray([date_positions[str(dates[index])] for index in valid_dates], dtype=np.int64)
    member_columns = np.asarray([asset_positions[value] for value in panel_assets], dtype=np.int64)
    eligible = np.load(panel_path / "eligible_mask.npy", mmap_mode="r", allow_pickle=False)
    factor_valid = np.load(panel_path / "factor_valid_mask.npy", mmap_mode="r", allow_pickle=False)
    daily_counts: list[int] = []
    for panel_row, member_row in zip(valid_dates, member_rows):
        members = membership_values[member_row, member_columns]
        eligible_row = np.asarray(eligible[panel_row], dtype=bool)
        has_factor = np.asarray(factor_valid[panel_row], dtype=bool).any(axis=0)
        daily_counts.append(int(np.count_nonzero(members & eligible_row & has_factor)))
    counts = np.asarray(daily_counts, dtype=np.float64)
    count_p01, count_p05 = np.quantile(counts, [0.01, 0.05])

    return {
        "statistics_period": {
            "start": str(dates[valid_dates[0]]),
            "end": str(dates[valid_dates[-1]]),
            "note": "Train plus validation only. No 2022 values were read for statistics.",
        },
        "daily_effective_csi500_stock_count": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
            "p01": float(count_p01),
            "p05": float(count_p05),
        },
        "factor_statistics": factor_stats,
        "execution_1d_return": execution_stats,
        "forward_returns": forward_stats,
    }


def _membership_metadata(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return item
    try:
        with np.load(path, allow_pickle=False) as archive:
            item["keys"] = list(archive.files)
            for key in ("dates", "asset_ids", "members"):
                if key in archive:
                    item[key] = {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
    except Exception as exc:
        item["read_error"] = f"{type(exc).__name__}: {exc}"
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/daily_strategy.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/daily_strategy_diagnostics/preflight_report.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    panel_path = Path(config.data.path or "")
    manifest_path = Path(config.data.factor_selection_manifest_path or "")
    directions_path = Path(config.data.factor_directions_path or "")
    paths = {
        "factor_panel": _panel_metadata(panel_path),
        "factor_selection_manifest": _json_metadata(manifest_path),
        "factor_directions": _json_metadata(directions_path),
        "factor_names_manifest_alignment": _factor_manifest_alignment(panel_path, manifest_path),
        "external_return_axis_alignment": _external_array_alignment(panel_path, config),
        "execution_1d_return": _array_metadata(Path(config.data.execution_return_path or "")),
        "forward_returns": _array_metadata(Path(config.data.return_path or "")),
        "csi500_membership": _membership_metadata(Path(config.data.membership_path or "")),
        "industry_data": _industry_metadata(Path(config.data.industry_npz_path or "")),
        "benchmark": _array_metadata(Path(config.data.benchmark_path or "")),
        "output_directory": {
            "path": str(Path(config.logging.output_dir)),
            "exists": Path(config.logging.output_dir).exists(),
        },
    }
    missing = [
        name
        for name, value in paths.items()
        if name != "output_directory" and not bool(value.get("exists"))
    ]

    # The configured labels use the panel date axis.  This documents alignment
    # without silently substituting an absent execution-return array.
    panel_dates = panel_path / "dates.npy"
    report: dict[str, Any] = {
        "status": "failed" if missing else "passed",
        "stop_before_training": bool(missing),
        "configured_daily_strategy": config.to_dict(),
        "core_paths": paths,
        "missing_core_inputs": missing,
        "date_ranges": {
            "factor_panel": _date_range(panel_dates),
            "execution_1d_return": _date_range(panel_dates)
            if not missing or "execution_1d_return" not in missing
            else None,
            "forward_returns": _date_range(panel_dates),
            "membership": _date_range(Path(config.data.membership_path or ""), "dates"),
            "industry": _date_range(Path(config.data.industry_npz_path or ""), "index"),
        },
        "split": {
            "train": [config.data.train_start, config.data.train_end],
            "validation": [config.data.validation_start, config.data.validation_end],
            "test": [config.data.test_start, config.data.test_end],
        },
        "full_statistics": None if missing else _full_statistics(config, panel_path),
        "statistics_note": (
            "Not computed because preflight is fail-closed: at least one configured "
            "core input is absent."
            if missing
            else "Computed on train plus validation only; no 2022 return values were read."
        ),
        "non_substituted_candidate": {
            "path": str(panel_path / "forward_returns.npy"),
            "exists": (panel_path / "forward_returns.npy").is_file(),
            "note": (
                "This candidate was not substituted for the missing configured "
                "execution_1d_return path."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
