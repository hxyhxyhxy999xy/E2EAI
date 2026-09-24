"""Build a train-only, IC-ranked, correlation-de-duplicated 64-factor panel.

Only the 710 purged 2018-01-02--2020-12-03 decision dates are allowed to
affect ranking, directions, or correlations.  The output panel separately
retains 2021 for later validation, but this builder never reads it for factor
selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from e2eai.config import load_config
from e2eai.data.correlation_dedup import (
    CorrelationResult,
    greedy_ic_then_correlation_dedup,
    mean_daily_cross_sectional_spearman,
)
from e2eai.data.train_only_selection import select_train_only_factors
from scripts.build_trainonly64_panel import (
    _actual_train_indices,
    _as_dates,
    _build_panel,
    _daily_rankic_worker,
    _decode,
    _git_commit,
    _init_worker,
    _pricing_base_arrays,
    _read_aligned_factor,
    _selection_universe,
    _sha256,
)


CANDIDATE_COUNT = 1245
TOP_K = 64
CORR_THRESHOLD = 0.80
MIN_COMMON_STOCKS = 30
MIN_COMMON_DATES = 100


def _load_factor_list(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("factor_names") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Invalid factor-name file: {path}")
    return values


def _load_directions(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("directions") if isinstance(payload, dict) and "directions" in payload else payload
    if not isinstance(values, dict):
        raise ValueError(f"Invalid direction file: {path}")
    return {str(name): int(value) for name, value in values.items()}


def _compact_factor_values(
    path: Path,
    decision_dates: pd.DatetimeIndex,
    panel_assets: np.ndarray,
    asset_positions: np.ndarray,
) -> np.ndarray:
    """Read one raw factor only on the daily CSI500 membership positions."""
    raw = _read_aligned_factor(path, decision_dates, panel_assets, dtype=np.float64)
    safe_positions = np.maximum(asset_positions, 0)
    compact = np.take_along_axis(raw, safe_positions, axis=1)
    return np.where(asset_positions >= 0, compact, np.nan)


def _factor_value_getter(
    candidate_paths: dict[str, Path],
    decision_dates: pd.DatetimeIndex,
    panel_assets: np.ndarray,
    asset_positions: np.ndarray,
    *,
    cache_size: int,
) -> Callable[[str], np.ndarray]:
    """Keep at most the current candidate plus selected compact factor arrays."""

    @functools.lru_cache(maxsize=cache_size)
    def load(name: str) -> np.ndarray:
        return _compact_factor_values(
            candidate_paths[name], decision_dates, panel_assets, asset_positions
        )

    return load


def _correlation_getter(
    load_values: Callable[[str], np.ndarray],
    base_mask: np.ndarray,
    *,
    cache_pairs: bool,
) -> Callable[[str, str], CorrelationResult]:
    cache: dict[tuple[str, str], CorrelationResult] = {}

    def get(left: str, right: str) -> CorrelationResult:
        key = tuple(sorted((left, right)))
        if cache_pairs and key in cache:
            return cache[key]
        result = mean_daily_cross_sectional_spearman(
            load_values(left),
            load_values(right),
            base_mask,
            min_common_stocks=MIN_COMMON_STOCKS,
            min_valid_dates=MIN_COMMON_DATES,
        )
        if cache_pairs:
            cache[key] = result
        return result

    return get


def _correlation_diagnostics(
    factor_names: list[str],
    get: Callable[[str, str], CorrelationResult],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Compute a complete valid 64x64 matrix plus auditable pair summaries."""
    size = len(factor_names)
    matrix = np.eye(size, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(factor_names):
        for right_index in range(left_index + 1, size):
            right = factor_names[right_index]
            result = get(left, right)
            if not result.valid:
                raise RuntimeError(f"Selected pair has invalid correlation support: {left}, {right}")
            matrix[left_index, right_index] = result.mean_correlation
            matrix[right_index, left_index] = result.mean_correlation
            rows.append(
                {
                    "factor_name_a": left,
                    "factor_name_b": right,
                    "mean_daily_cross_sectional_spearman": result.mean_correlation,
                    "abs_mean_daily_cross_sectional_spearman": abs(result.mean_correlation),
                    "valid_date_count": result.valid_date_count,
                    "min_common_stock_count": result.min_common_stock_count,
                }
            )
    pairs = pd.DataFrame(rows).sort_values(
        ["abs_mean_daily_cross_sectional_spearman", "factor_name_a", "factor_name_b"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    abs_values = pairs["abs_mean_daily_cross_sectional_spearman"].to_numpy(dtype=float)
    summary = {
        "factor_count": int(size),
        "pair_count": int(len(pairs)),
        "max_pair_abs_corr": float(abs_values.max()) if len(abs_values) else 0.0,
        "median_pair_abs_corr": float(np.median(abs_values)) if len(abs_values) else 0.0,
        "mean_pair_abs_corr": float(abs_values.mean()) if len(abs_values) else 0.0,
        "pair_count_abs_corr_gt_0_5": int((abs_values > 0.5).sum()),
        "pair_count_abs_corr_gt_0_6": int((abs_values > 0.6).sum()),
        "pair_count_abs_corr_gt_0_7": int((abs_values > 0.7).sum()),
        "pair_count_abs_corr_ge_0_8": int((abs_values >= 0.8).sum()),
        "correlation_metric": "mean_daily_cross_sectional_spearman",
        "correlation_threshold": CORR_THRESHOLD,
        "min_common_stocks": MIN_COMMON_STOCKS,
        "min_valid_dates": MIN_COMMON_DATES,
    }
    return pd.DataFrame(matrix, index=factor_names, columns=factor_names), summary, pairs


def _yearly_rankic_statistics(
    statistics: pd.DataFrame,
    daily_rankic: np.ndarray,
    decision_dates: pd.DatetimeIndex,
    ordered_names: list[str],
) -> pd.DataFrame:
    values = statistics.copy()
    for year in (2018, 2019, 2020):
        rows = np.asarray(decision_dates.year == year)
        selected = daily_rankic[rows]
        means = np.full(selected.shape[1], np.nan, dtype=float)
        for index in range(selected.shape[1]):
            finite = selected[:, index][np.isfinite(selected[:, index])]
            if finite.size:
                means[index] = float(finite.mean())
        by_name = dict(zip(ordered_names, means))
        values[f"rankic_{year}"] = values["factor_name"].map(by_name)
    annual = values[["rankic_2018", "rankic_2019", "rankic_2020"]].to_numpy(dtype=float)
    signs = np.sign(annual)
    nonzero = np.all(np.isfinite(annual) & (signs != 0.0), axis=1)
    values["three_years_same_direction"] = nonzero & (np.abs(signs.sum(axis=1)) == 3)
    values["annual_sign_flip"] = nonzero & (np.abs(signs.sum(axis=1)) != 3)
    return values


def _write_report(
    path: Path,
    *,
    old_summary: dict[str, Any],
    new_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    old_selected: pd.DataFrame,
    new_selected: pd.DataFrame,
) -> None:
    old_abs = float(old_selected["abs_train_mean_rankic"].mean())
    new_abs = float(new_selected["abs_train_mean_rankic"].mean())
    lines = [
        "# Train-only corr08 64-factor selection report",
        "",
        "## Scope",
        "",
        "Selection, directions, and correlations use only 710 actual purged training decision dates "
        "from 2018-01-02 through 2020-12-03. The target is VWAP[t+2]/VWAP[t+1]-1. "
        "No 2021 RankIC/correlation and no 2022 data were used.",
        "",
        "## Redundancy comparison",
        "",
        "| Metric | Old IC Top64 | New IC + corr08 64 | Change (new − old) |",
        "|---|---:|---:|---:|",
        f"| Max pair |corr| | {old_summary['max_pair_abs_corr']:.4f} | {new_summary['max_pair_abs_corr']:.4f} | {new_summary['max_pair_abs_corr'] - old_summary['max_pair_abs_corr']:.4f} |",
        f"| Median pair |corr| | {old_summary['median_pair_abs_corr']:.4f} | {new_summary['median_pair_abs_corr']:.4f} | {new_summary['median_pair_abs_corr'] - old_summary['median_pair_abs_corr']:.4f} |",
        f"| Mean pair |corr| | {old_summary['mean_pair_abs_corr']:.4f} | {new_summary['mean_pair_abs_corr']:.4f} | {new_summary['mean_pair_abs_corr'] - old_summary['mean_pair_abs_corr']:.4f} |",
        f"| Pairs |corr| > 0.5 | {old_summary['pair_count_abs_corr_gt_0_5']} | {new_summary['pair_count_abs_corr_gt_0_5']} | {new_summary['pair_count_abs_corr_gt_0_5'] - old_summary['pair_count_abs_corr_gt_0_5']} |",
        f"| Pairs |corr| > 0.7 | {old_summary['pair_count_abs_corr_gt_0_7']} | {new_summary['pair_count_abs_corr_gt_0_7']} | {new_summary['pair_count_abs_corr_gt_0_7'] - old_summary['pair_count_abs_corr_gt_0_7']} |",
        f"| Pairs |corr| >= 0.8 | {old_summary['pair_count_abs_corr_ge_0_8']} | {new_summary['pair_count_abs_corr_ge_0_8']} | {new_summary['pair_count_abs_corr_ge_0_8'] - old_summary['pair_count_abs_corr_ge_0_8']} |",
        f"| Mean selected |train RankIC| | {old_abs:.6f} | {new_abs:.6f} | {new_abs - old_abs:.6f} |",
        "",
        "## Interpretation",
        "",
        f"- The corr08 rule replaces {comparison_summary['removed_count']} old factors and adds {comparison_summary['added_count']} factors; overlap is {comparison_summary['overlap_count']}/64.",
        "- A lower pair-correlation summary indicates less direct redundancy under the stipulated daily cross-sectional definition. It does not prove incremental return or validation alpha.",
        "- The selected set keeps raw processed factor values; no PCA, residualisation, orthogonalisation, or direction multiplication was applied to the panel.",
        "- If the correlation cap is satisfied, the corr08 panel is a better-diversified *candidate* for a later, separately authorised 2021 validation experiment. This report makes no 2021-performance claim.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/daily_strategy.yaml"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("/cloud/factor_synthesis/factors/h5"))
    parser.add_argument("--pricing-h5", type=Path, default=Path("/cloud/hdf5/historical/china_astock_2018.h5"))
    parser.add_argument("--selection-output", type=Path, default=Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08"))
    parser.add_argument("--panel-output", type=Path, default=Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08_panel"))
    parser.add_argument("--report-output", type=Path, default=Path("output/daily_strategy_diagnostics/corr08_selection_report.md"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--block-dates", type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1 or args.block_dates < 1:
        parser.error("workers and block-dates must be positive")
    if args.selection_output.exists() or args.panel_output.exists():
        raise FileExistsError("Refusing to overwrite existing corr08 selection or panel output")

    config = load_config(args.config)
    source_panel = Path(config.data.path or "")
    source_dates = _as_dates(np.load(source_panel / "dates.npy", allow_pickle=False))
    source_assets = _decode(np.load(source_panel / "asset_ids.npy", allow_pickle=False))
    train_indices, selection_dates = _actual_train_indices(config, source_dates)
    actual_dates = source_dates[train_indices]
    if len(actual_dates) != 710 or str(actual_dates[0].date()) != "2018-01-02" or str(actual_dates[-1].date()) != "2020-12-03":
        raise RuntimeError(f"Unexpected permitted selection dates: {selection_dates}")

    candidate_paths = {path.stem: path for path in sorted(args.candidate_dir.glob("*.h5"))}
    if len(candidate_paths) != CANDIDATE_COUNT:
        raise RuntimeError(f"Expected exactly {CANDIDATE_COUNT} H5 candidates, found {len(candidate_paths)}")
    args.selection_output.mkdir(parents=True, exist_ok=False)
    candidate_names = sorted(candidate_paths)
    (args.selection_output / "candidate_factor_names.json").write_text(
        json.dumps({"candidate_count": len(candidate_names), "factor_names": candidate_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.selection_output / "factor_selection_train_dates.json").write_text(
        json.dumps(selection_dates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    execution_source = Path(config.data.execution_return_path or "")
    execution = np.load(execution_source, mmap_mode="r", allow_pickle=False)
    pricing_train = _pricing_base_arrays(
        args.pricing_h5, actual_dates, source_assets, block_dates=args.block_dates
    )
    positions, selected_execution, base_mask = _selection_universe(
        actual_dates,
        source_assets,
        pricing_train["base_tradable"],
        np.asarray(execution[train_indices], dtype=np.float64),
        Path(config.data.membership_path or ""),
    )
    context = {
        "decision_dates": actual_dates.to_numpy(),
        "panel_assets": source_assets,
        "asset_positions": positions,
        "execution_values": selected_execution,
        "base_universe_mask": base_mask,
    }
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(context,)
    ) as executor:
        futures = {executor.submit(_daily_rankic_worker, str(path)): name for name, path in candidate_paths.items()}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"rankic_candidates={completed}/{len(futures)}", flush=True)
    results.sort(key=lambda row: row["factor_name"])
    ordered_names = [str(row["factor_name"]) for row in results]
    daily_rankic = np.column_stack([row["daily_rankic"] for row in results])
    statistics = select_train_only_factors(
        ordered_names, daily_rankic, np.ones(len(actual_dates), dtype=bool), top_k=TOP_K
    )
    statistics["rank_by_abs_ic"] = statistics["rank"].astype(int)
    statistics = _yearly_rankic_statistics(statistics, daily_rankic, actual_dates, ordered_names)
    sources = pd.DataFrame(
        {
            "factor_name": ordered_names,
            "candidate_factor_source_path": [row["path"] for row in results],
            "read_error": [row["error"] for row in results],
        }
    )
    statistics = statistics.merge(sources, on="factor_name", how="left", validate="one_to_one")

    load_values = _factor_value_getter(
        candidate_paths, actual_dates, source_assets, positions, cache_size=TOP_K + 1
    )
    pair_getter = _correlation_getter(load_values, base_mask, cache_pairs=False)
    selection_audit, selected_names = greedy_ic_then_correlation_dedup(
        statistics, pair_getter, top_k=TOP_K, correlation_threshold=CORR_THRESHOLD
    )
    selected = selection_audit.loc[selection_audit["accepted"]].sort_values("selected_rank")
    if len(selected) != TOP_K or selected["train_mean_rankic"].isna().any():
        raise RuntimeError("Greedy corr08 selection did not produce 64 finite factors")
    selection_path = args.selection_output / "factor_selection_corr08.csv"
    selection_audit.to_csv(selection_path, index=False, encoding="utf-8-sig")
    directions = {
        str(row.factor_name): int(-1 if float(row.train_mean_rankic) < 0.0 else 1)
        for row in selected.itertuples(index=False)
    }
    direction_path = args.selection_output / "factor_directions.json"
    direction_path.write_text(json.dumps(directions, ensure_ascii=False, indent=2), encoding="utf-8")
    factor_list_path = args.selection_output / "factor_list.json"
    factor_list_path.write_text(
        json.dumps({"factor_names": selected_names}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_load = _factor_value_getter(
        candidate_paths, actual_dates, source_assets, positions, cache_size=TOP_K + 1
    )
    selected_get = _correlation_getter(selected_load, base_mask, cache_pairs=True)
    new_matrix, new_summary, new_pairs = _correlation_diagnostics(selected_names, selected_get)
    if new_summary["pair_count_abs_corr_ge_0_8"] != 0:
        raise RuntimeError("corr08 selected set violates the strict abs(corr) < 0.80 rule")
    new_matrix_path = args.selection_output / "selected64_correlation_matrix.csv"
    new_pairs_path = args.selection_output / "selected64_highest_corr_pairs.csv"
    new_summary_path = args.selection_output / "selected64_correlation_summary.json"
    new_matrix.to_csv(new_matrix_path, encoding="utf-8-sig")
    new_pairs.head(20).to_csv(new_pairs_path, index=False, encoding="utf-8-sig")
    new_summary_path.write_text(json.dumps(new_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep the comparison input explicit and on the current formal panel.
    # The retired pre-dedup panel is intentionally not referenced by code.
    old_root = Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08")
    old_names = _load_factor_list(old_root / "factor_list.json")
    old_directions = _load_directions(old_root / "factor_directions.json")
    if len(old_names) != TOP_K:
        raise RuntimeError("Old train-only selection is not a 64-factor list")
    old_load = _factor_value_getter(
        candidate_paths, actual_dates, source_assets, positions, cache_size=TOP_K + 1
    )
    old_get = _correlation_getter(old_load, base_mask, cache_pairs=True)
    old_matrix, old_summary, old_pairs = _correlation_diagnostics(old_names, old_get)
    old_matrix.to_csv(args.selection_output / "reference_correlation_matrix.csv", encoding="utf-8-sig")
    old_pairs.head(20).to_csv(args.selection_output / "reference_highest_corr_pairs.csv", index=False, encoding="utf-8-sig")
    (args.selection_output / "reference_correlation_summary.json").write_text(
        json.dumps(old_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    old_rank_file = pd.read_csv(old_root / "factor_selection.csv", encoding="utf-8-sig")
    old_rank = {str(row.factor_name): int(row["rank"]) for _, row in old_rank_file.iterrows()}
    new_rank = {str(row.factor_name): int(row.selected_rank) for row in selected.itertuples(index=False)}
    lookup = selection_audit.set_index("factor_name")
    comparison = pd.DataFrame(
        {
            "factor_name": candidate_names,
            "in_reference_selection": [name in old_names for name in candidate_names],
            "in_new_corr08_64": [name in new_rank for name in candidate_names],
            "reference_rank": [old_rank.get(name) for name in candidate_names],
            "new_rank_if_selected": [new_rank.get(name) for name in candidate_names],
            "train_mean_rankic": [lookup.at[name, "train_mean_rankic"] for name in candidate_names],
            "new_direction": [directions.get(name) for name in candidate_names],
            "reference_direction_if_available": [old_directions.get(name) for name in candidate_names],
        }
    )
    comparison_path = args.selection_output / "factor_selection_reference_vs_corr08.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    comparison_summary = {
        "overlap_count": int((comparison["in_reference_selection"] & comparison["in_new_corr08_64"]).sum()),
        "added_count": int((~comparison["in_reference_selection"] & comparison["in_new_corr08_64"]).sum()),
        "removed_count": int((comparison["in_reference_selection"] & ~comparison["in_new_corr08_64"]).sum()),
        "old_correlation_summary": old_summary,
        "new_correlation_summary": new_summary,
        "new_selected_three_years_same_direction": int(selected["three_years_same_direction"].sum()),
        "new_selected_annual_sign_flip": int(selected["annual_sign_flip"].sum()),
    }
    comparison_summary_path = args.selection_output / "factor_selection_reference_vs_corr08_summary.json"
    comparison_summary_path.write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2), encoding="utf-8"
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
    panel_metadata_path = args.panel_output / "metadata.json"
    panel_metadata = json.loads(panel_metadata_path.read_text(encoding="utf-8"))
    panel_metadata.update(
        {
            "schema": "daily_strategy_trainonly64_corr08_v1",
            "correlation_filter": "abs(mean_daily_cross_sectional_spearman) < 0.80",
            "correlation_threshold": CORR_THRESHOLD,
            "correlation_min_common_stocks": MIN_COMMON_STOCKS,
            "correlation_min_valid_dates": MIN_COMMON_DATES,
            "selection_algorithm": "greedy_ic_then_correlation_dedup",
            "selection_manifest": str(args.selection_output / "factor_selection_manifest.json"),
        }
    )
    panel_metadata_path.write_text(json.dumps(panel_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_report(
        args.report_output,
        old_summary=old_summary,
        new_summary=new_summary,
        comparison_summary=comparison_summary,
        old_selected=selection_audit.loc[selection_audit["factor_name"].isin(old_names)],
        new_selected=selected,
    )
    key_paths = {
        "candidate_factor_names.json": args.selection_output / "candidate_factor_names.json",
        "factor_selection_train_dates.json": args.selection_output / "factor_selection_train_dates.json",
        "factor_selection_corr08.csv": selection_path,
        "factor_list.json": factor_list_path,
        "factor_directions.json": direction_path,
        "selected64_correlation_matrix.csv": new_matrix_path,
        "selected64_correlation_summary.json": new_summary_path,
        "selected64_highest_corr_pairs.csv": new_pairs_path,
        "factor_selection_reference_vs_corr08.csv": comparison_path,
        "factor_selection_reference_vs_corr08_summary.json": comparison_summary_path,
        "panel_metadata.json": panel_metadata_path,
        "panel_alpha_names.npy": args.panel_output / "alpha_names.npy",
        "panel_alpha_tensor.npy": args.panel_output / "alpha_tensor.npy",
        "corr08_selection_report.md": args.report_output,
    }
    manifest = {
        "schema": "train_only_corr08_factor_selection_v1",
        "candidate_count": len(candidate_paths),
        "selected_count": len(selected_names),
        "candidates_traversed": int(
            selection_audit["accepted"].sum()
            + (selection_audit["rejection_reason"] != "not_evaluated_after_target_reached")
            .fillna(False)
            .sum()
        ),
        "correlation_rejection_count": int(
            (selection_audit["rejection_reason"] == "correlation_threshold").sum()
        ),
        "invalid_correlation_rejection_count": int(
            (selection_audit["rejection_reason"] == "invalid_correlation_to_selected").sum()
        ),
        "selection_period_start": selection_dates["actual_train_start"],
        "selection_period_end": selection_dates["actual_train_end"],
        "n_selection_dates": selection_dates["n_train_dates"],
        "selection_target": "VWAP[t+2]/VWAP[t+1]-1",
        "ranking_metric": "abs(mean_daily_cross_sectional_spearman_rankic)",
        "correlation_metric": "mean_daily_cross_sectional_spearman",
        "correlation_filter": "abs(mean_daily_cross_sectional_spearman) < 0.80",
        "correlation_threshold": CORR_THRESHOLD,
        "min_common_stocks": MIN_COMMON_STOCKS,
        "min_valid_dates": MIN_COMMON_DATES,
        "selection_algorithm": "greedy_ic_then_correlation_dedup",
        "direction_definition": "sign(train_mean_rankic), exact zero -> +1",
        "factor_list_path": str(factor_list_path),
        "factor_directions_path": str(direction_path),
        "factor_selection_path": str(selection_path),
        "panel_path": str(args.panel_output),
        "factor_names": selected_names,
        "file_sha256": {name: _sha256(path) for name, path in key_paths.items()},
        "creation_time_utc": datetime.now(UTC).isoformat(),
        "code_version_git_commit": _git_commit(Path.cwd()),
        "lookahead_safe": True,
        "upstream_warning": "PASS_WITH_UPSTREAM_WARNING: raw H5 formula/publication-time provenance requires separate audit.",
    }
    manifest_path = args.selection_output / "factor_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": len(candidate_paths),
                "selection_dates": [selection_dates["actual_train_start"], selection_dates["actual_train_end"]],
                "selected_count": len(selected_names),
                "candidates_traversed": int(
                    selection_audit["accepted"].sum()
                    + (selection_audit["rejection_reason"] != "not_evaluated_after_target_reached")
                    .fillna(False)
                    .sum()
                ),
                "correlation_rejections": int((selection_audit["rejection_reason"] == "correlation_threshold").sum()),
                "selection_output": str(args.selection_output),
                "panel_output": str(args.panel_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
