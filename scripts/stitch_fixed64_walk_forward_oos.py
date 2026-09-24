"""Stitch fixed-64 annual-fold OOS targets with one continuous portfolio state."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2eai.config import load_config
from e2eai.data.loaders import build_dataloaders
from e2eai.evaluation.metrics import maximum_drawdown
from e2eai.models.baselines import factor_mean_scores
from e2eai.training.net_return import numpy_drifted_rows
from e2eai.workflows import load_configured_data
from scripts.run_fixed64_walk_forward import COST_BPS, FOLDS, _canonical_dates, _json, _load_metadata, _make_fold_config, _root


def _annualized(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    wealth = float(np.prod(1.0 + values))
    return wealth ** (252.0 / len(values)) - 1.0 if wealth > 0 else -1.0


def _sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=0))
    return float(np.sqrt(252.0) * values.mean() / std) if std > 1e-12 else 0.0


def _path_metrics(
    gross: np.ndarray,
    turnover: np.ndarray,
    cost: np.ndarray,
    net: np.ndarray,
    benchmark: np.ndarray,
    effective: np.ndarray,
    maximum_weight: np.ndarray,
) -> dict[str, float]:
    active = net - benchmark
    active_std = float(active.std(ddof=0))
    benchmark_annual = _annualized(benchmark)
    return {
        "return_0bp_annualized": _annualized(gross),
        "return_7_5bp_annualized": _annualized(net),
        "return_7_5bp_cumulative": float(np.prod(1.0 + net) - 1.0),
        "sharpe_7_5bp": _sharpe(net),
        "mdd_7_5bp": float(maximum_drawdown(net)),
        "mean_one_way_turnover": float(np.mean(turnover)),
        "median_turnover": float(np.median(turnover)),
        "p95_turnover": float(np.quantile(turnover, 0.95)),
        "annual_transaction_cost_drag": _annualized(gross) - _annualized(net),
        "cumulative_transaction_cost_drag": float(np.sum(cost)),
        "mean_effective_n": float(np.mean(effective)),
        "mean_max_weight": float(np.mean(maximum_weight)),
        "absolute_max_weight": float(np.max(maximum_weight)),
        "csi500_return": benchmark_annual,
        "annual_excess_return": _annualized(net) - benchmark_annual,
        "tracking_error": float(np.sqrt(252.0) * active_std),
        "information_ratio": float(np.sqrt(252.0) * active.mean() / active_std) if active_std > 1e-12 else 0.0,
        "active_mdd": float(maximum_drawdown(active)),
        "active_win_rate": float((active > 0).mean()),
        "cumulative_active_return": float(np.prod(1.0 + active) - 1.0),
    }


def _frame_to_maps(frame: pd.DataFrame) -> tuple[list[str], list[dict[str, float]], list[dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    dates: list[str] = []
    targets: list[dict[str, float]] = []
    returns: list[dict[str, float]] = []
    effective: list[float] = []
    max_weight: list[float] = []
    benchmark: list[float] = []
    for date, group in frame.groupby("date", sort=True):
        selected = group.loc[group["valid_asset"].astype(bool)].copy()
        weights = pd.to_numeric(selected["target_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        good = np.isfinite(weights) & (weights > 0)
        values = weights[good]
        if not len(values):
            raise ValueError(f"no selected target weights on {date}")
        dates.append(str(date))
        targets.append({str(row.asset): float(row.target_weight) for row in selected.itertuples(index=False) if np.isfinite(row.target_weight) and float(row.target_weight) > 0})
        returns.append({str(row.asset): float(row.execution_return) for row in group.itertuples(index=False) if np.isfinite(row.execution_return)})
        effective.append(float(1.0 / np.square(values).sum()))
        max_weight.append(float(values.max()))
        values_benchmark = pd.to_numeric(group.get("csi500_return"), errors="coerce").dropna().unique()
        if len(values_benchmark) != 1:
            raise ValueError(f"benchmark must be one finite value per date: {date}")
        benchmark.append(float(values_benchmark[0]))
    return dates, targets, returns, np.asarray(effective), np.asarray(max_weight), np.asarray(benchmark)


def _load_mlp_targets(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for fold in FOLDS:
        target = root / f"fold_{fold}" / "oos_target_weights.parquet"
        manifest = root / f"fold_{fold}" / "fold_manifest.json"
        if not target.is_file() or not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError(f"all folds must PASS before formal stitching; missing or failed fold {fold}")
        frame = pd.read_parquet(target)
        frame["fold"] = fold
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    return result


def _factor_mean_frame(base_config: Path, root: Path) -> pd.DataFrame:
    base = load_config(base_config)
    metadata = _load_metadata(base)
    config = _make_fold_config(base, 0, root)
    config.data.train_start = "2018-01-01"; config.data.train_end = "2020-12-31"
    config.data.validation_start = "2021-01-01"; config.data.validation_end = "2021-12-31"
    config.data.test_start = "2022-01-01"; config.data.test_end = str(metadata["latest_legal_execution_date"])
    config.data.apply_factor_direction = True
    source, _ = load_configured_data(config)
    loaders = build_dataloaders(config, source)
    rows: list[dict[str, Any]] = []
    for batch in loaders.test:
        factors = batch["raw_factors"]
        masks = batch["asset_mask"].bool()
        scores = factor_mean_scores(factors, masks).detach().cpu().numpy()
        returns = batch["execution_1d_return"].detach().cpu().numpy()
        return_masks = batch["execution_return_mask"].detach().cpu().numpy().astype(bool)
        for index, date in enumerate(batch["dates"]):
            assets = [str(asset) for asset in batch["asset_ids"][index][: int(masks[index].sum())]]
            valid = masks[index, : len(assets)].detach().cpu().numpy().astype(bool)
            raw = scores[index, : len(assets)]
            order = pd.DataFrame({"asset": assets, "raw_score": raw, "valid": valid}).query("valid").sort_values(["raw_score", "asset"], ascending=[False, True], kind="stable")
            k = max(1, int(np.ceil(0.20 * len(order))))
            chosen = set(order.iloc[:k]["asset"].tolist())
            for position, asset in enumerate(assets):
                rows.append({"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "asset": asset, "raw_score": float(raw[position]), "normalized_score": np.nan, "target_weight": 1.0 / k if asset in chosen else 0.0, "valid_asset": asset in chosen, "execution_return": float(returns[index, position]) if return_masks[index, position] else np.nan})
    result = pd.DataFrame(rows)
    # Same execution benchmark used by MLP, attached once per decision date.
    benchmark = source.benchmark_returns
    lookup = {pd.Timestamp(date).strftime("%Y-%m-%d"): float(value) for date, value in zip(source.dates, benchmark)}
    result["csi500_return"] = result["date"].map(lookup)
    out = root / "factor_mean"; out.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out / "factor_mean_target_weights.parquet", index=False)
    _json(out / "factor_mean_manifest.json", {"status": "PASS", "trained": False, "factor_directions_applied": True, "definition": "mean(frozen_direction * fixed_factor), daily Top20% equal-weight", "date_range": [str(result.date.min()), str(result.date.max())], "date_count": int(result.date.nunique())})
    return result


def _attach_benchmark(frame: pd.DataFrame, base_config: Path, root: Path) -> pd.DataFrame:
    if "csi500_return" in frame.columns:
        return frame
    base = load_config(base_config)
    metadata = _load_metadata(base)
    config = _make_fold_config(base, 0, root)
    config.data.test_start = "2022-01-01"; config.data.test_end = str(metadata["latest_legal_execution_date"])
    source, _ = load_configured_data(config)
    lookup = {pd.Timestamp(date).strftime("%Y-%m-%d"): float(value) for date, value in zip(source.dates, source.benchmark_returns)}
    result = frame.copy(); result["csi500_return"] = result["date"].map(lookup)
    if result["csi500_return"].isna().any():
        raise ValueError("CSI500 execution benchmark is missing stitched decision dates")
    return result


def _strict_calendar_check(dates: list[str], base_config: Path, root: Path) -> dict[str, Any]:
    base = load_config(base_config)
    metadata = _load_metadata(base)
    config = _make_fold_config(base, 0, root)
    config.data.test_start = "2022-01-01"; config.data.test_end = str(metadata["latest_legal_execution_date"])
    source, _ = load_configured_data(config)
    universe = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in source.dates]
    lookup = {date: index for index, date in enumerate(universe)}
    position = [lookup[date] for date in dates]
    unexpected = [(dates[index - 1], dates[index]) for index in range(1, len(position)) if position[index] != position[index - 1] + 1]
    return {"status": "PASS" if not unexpected else "FAIL", "strict_monotonic": all(dates[index] < dates[index + 1] for index in range(len(dates) - 1)), "unique": len(dates) == len(set(dates)), "unexpected_calendar_gaps": unexpected[:20]}


def _make_daily(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates, targets, returns, effective, max_weight, benchmark = _frame_to_maps(frame)
    _, gross, turnover, cost, net = numpy_drifted_rows(targets, returns)
    daily = pd.DataFrame({"date": dates, f"{label}_gross_return": gross, f"{label}_turnover": turnover, f"{label}_cost_7_5bps": cost, f"{label}_net_return_7_5bps": net, f"{label}_effective_n": effective, f"{label}_max_weight": max_weight, "csi500_return": benchmark})
    return daily, {"dates": dates, "gross": gross, "turnover": turnover, "cost": cost, "net": net, "effective": effective, "max_weight": max_weight, "benchmark": benchmark}


def _annual_rows(label: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    dates = np.asarray(state["dates"])
    years = [2022, 2023, 2024, 2025, 2026]
    rows: list[dict[str, Any]] = []
    for year in years:
        mask = np.asarray([pd.Timestamp(str(value)).year == year for value in dates])
        if not mask.any():
            continue
        metrics = _path_metrics(state["gross"][mask], state["turnover"][mask], state["cost"][mask], state["net"][mask], state["benchmark"][mask], state["effective"][mask], state["max_weight"][mask])
        rows.append({"year": f"{year} YTD" if year == 2026 else str(year), "strategy": label, "start_date": str(dates[mask][0]), "end_date": str(dates[mask][-1]), "decision_dates": int(mask.sum()), **metrics})
    return rows


def _markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    table = frame[columns].copy()
    return table.to_markdown(index=False, floatfmt=".6f") if len(table) else "(no rows)"


def stitch(base_config: Path, output: Path) -> Path:
    root = _root(output)
    preflight = root / "data_preflight" / "preflight.json"
    if not preflight.is_file() or json.loads(preflight.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("cannot stitch before the fixed64 preflight PASS")
    destination = root / "stitched"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing stitched output: {destination}")
    destination.mkdir(parents=True)
    mlp = _attach_benchmark(_load_mlp_targets(root), base_config, root)
    factor = _factor_mean_frame(base_config, root)
    mlp_dates = sorted(mlp["date"].unique().tolist())
    factor_dates = sorted(factor["date"].unique().tolist())
    calendar = _strict_calendar_check(mlp_dates, base_config, root)
    if calendar["status"] != "PASS" or factor_dates != mlp_dates:
        raise RuntimeError("stitched date path is not unique, monotonic, continuous, or factor-aligned")
    mlp_daily, mlp_state = _make_daily(mlp, "baseline")
    factor_daily, factor_state = _make_daily(factor, "factor_mean")
    daily = mlp_daily.merge(
        factor_daily.drop(columns=["csi500_return"]),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    daily["baseline_active_return"] = daily["baseline_net_return_7_5bps"] - daily["csi500_return"]
    daily["factor_mean_active_return"] = daily["factor_mean_net_return_7_5bps"] - daily["csi500_return"]
    daily.to_csv(destination / "stitched_oos_daily_returns.csv", index=False, encoding="utf-8-sig")
    equity = pd.DataFrame({"date": daily["date"], "baseline_mlp_nav": np.cumprod(1.0 + daily["baseline_net_return_7_5bps"]), "factor_mean_nav": np.cumprod(1.0 + daily["factor_mean_net_return_7_5bps"]), "csi500_nav": np.cumprod(1.0 + daily["csi500_return"]), "baseline_active_nav": np.cumprod(1.0 + daily["baseline_active_return"]), "factor_mean_active_nav": np.cumprod(1.0 + daily["factor_mean_active_return"])})
    equity.to_csv(destination / "stitched_oos_equity.csv", index=False, encoding="utf-8-sig")
    annual = pd.DataFrame([*_annual_rows("Baseline MLP", mlp_state), *_annual_rows("Factor Mean", factor_state)])
    annual.to_csv(destination / "oos_annual_comparison.csv", index=False, encoding="utf-8-sig")
    (destination / "oos_annual_comparison.md").write_text("# Stitched OOS annual comparison\n\n" + _markdown(annual, ["year", "strategy", "return_0bp_annualized", "return_7_5bp_annualized", "sharpe_7_5bp", "mdd_7_5bp", "mean_one_way_turnover", "mean_effective_n", "mean_max_weight", "csi500_return", "annual_excess_return", "tracking_error", "information_ratio"]) + "\n", encoding="utf-8")
    deltas: list[dict[str, Any]] = []
    for year in annual["year"].drop_duplicates():
        a = annual[(annual.year == year) & (annual.strategy == "Baseline MLP")].iloc[0]
        b = annual[(annual.year == year) & (annual.strategy == "Factor Mean")].iloc[0]
        deltas.append({"year": year, "delta_0bp_return": a.return_0bp_annualized - b.return_0bp_annualized, "delta_7_5bp_return": a.return_7_5bp_annualized - b.return_7_5bp_annualized, "delta_7_5bp_sharpe": a.sharpe_7_5bp - b.sharpe_7_5bp, "delta_mdd": a.mdd_7_5bp - b.mdd_7_5bp, "delta_turnover": a.mean_one_way_turnover - b.mean_one_way_turnover, "delta_effective_n": a.mean_effective_n - b.mean_effective_n, "delta_annual_excess": a.annual_excess_return - b.annual_excess_return, "delta_tracking_error": a.tracking_error - b.tracking_error, "delta_information_ratio": a.information_ratio - b.information_ratio})
    delta = pd.DataFrame(deltas)
    delta.to_csv(destination / "model_vs_factor_mean_oos_annual.csv", index=False, encoding="utf-8-sig")
    (destination / "model_vs_factor_mean_oos_annual.md").write_text("# Baseline MLP minus Factor Mean\n\n" + _markdown(delta, list(delta.columns)) + "\n", encoding="utf-8")
    switch_rows: list[dict[str, Any]] = []
    for year in (2023, 2024, 2025, 2026):
        mask = pd.to_datetime(daily["date"]).dt.year.eq(year)
        if mask.any():
            row = daily.loc[mask].iloc[0]
            switch_rows.append({"switch": f"{year - 1}->{year}", "decision_date": row.date, "one_way_turnover": row.baseline_turnover, "gross_return": row.baseline_gross_return, "cost_7_5bps": row.baseline_cost_7_5bps, "net_return_7_5bps": row.baseline_net_return_7_5bps})
    switches = pd.DataFrame(switch_rows)
    if len(switches) != 4 or not bool((switches["one_way_turnover"] > 0).all()):
        raise RuntimeError("model switch turnover was reset or a switch date is missing")
    switches.to_csv(destination / "model_switch_turnover.csv", index=False, encoding="utf-8-sig")
    overall = {"Baseline MLP": _path_metrics(mlp_state["gross"], mlp_state["turnover"], mlp_state["cost"], mlp_state["net"], mlp_state["benchmark"], mlp_state["effective"], mlp_state["max_weight"]), "Factor Mean": _path_metrics(factor_state["gross"], factor_state["turnover"], factor_state["cost"], factor_state["net"], factor_state["benchmark"], factor_state["effective"], factor_state["max_weight"]), "CSI500": {"annualized_return": _annualized(mlp_state["benchmark"]), "sharpe": _sharpe(mlp_state["benchmark"]), "mdd": float(maximum_drawdown(mlp_state["benchmark"])), "cumulative_return": float(np.prod(1.0 + mlp_state["benchmark"]) - 1.0)}}
    _json(destination / "stitched_oos_metrics.json", {"status": "PASS", "date_range": [mlp_dates[0], mlp_dates[-1]], "decision_date_count": len(mlp_dates), "calendar": calendar, "model_switch_turnover_continuous": True, "baseline": overall["Baseline MLP"], "factor_mean": overall["Factor Mean"], "csi500": overall["CSI500"]})
    fold_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        location = root / f"fold_{fold}"
        manifest = json.loads((location / "fold_manifest.json").read_text(encoding="utf-8"))
        validation = json.loads((location / "validation_metrics.json").read_text(encoding="utf-8"))
        isolated = json.loads((location / "oos_metrics_isolated.json").read_text(encoding="utf-8"))
        audit = json.loads((location / "full_period_rankic_audit.json").read_text(encoding="utf-8"))
        date_info = manifest["fold_dates"]
        fold_rows.append({"fold": fold, "train_start": date_info["used"]["train"][0], "train_end": date_info["used"]["train"][1], "validation_start": date_info["used"]["validation"][0], "validation_end": date_info["used"]["validation"][1], "oos_start": date_info["used"]["test"][0], "oos_end": date_info["used"]["test"][1], "a_best": manifest["stage_a"]["best_epoch"], "c_best": manifest["stage_c"]["best_optimizer_step"], "d_stability_best": manifest["stage_d_stability"]["best_optimizer_step"], "full_train_rankic": audit["full_train_rankic"], "full_validation_rankic": audit["full_validation_rankic"], "true_rankic_gap": audit["true_rankic_gap"], "validation_7_5bp_return": validation.get("validation_daily_return_cost_7p5bps"), "validation_sharpe": validation.get("validation_daily_sharpe_cost_7p5bps"), "validation_turnover": validation.get("validation_daily_turnover"), "validation_ir": validation.get("validation_information_ratio_cost_7p5bps"), "oos_isolated_7_5bp_return_DIAGNOSTIC_ONLY": isolated.get("validation_daily_return_7_5bps"), "oos_isolated_sharpe_DIAGNOSTIC_ONLY": isolated.get("validation_daily_sharpe_7_5bps"), "oos_isolated_turnover_DIAGNOSTIC_ONLY": isolated.get("validation_daily_turnover")})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(destination / "fold_summary.csv", index=False, encoding="utf-8-sig")
    positive = annual[annual.strategy == "Baseline MLP"]
    diagnosis = {"baseline_positive_net_years": int((positive.return_7_5bp_annualized > 0).sum()), "baseline_positive_excess_years": int((positive.annual_excess_return > 0).sum()), "baseline_positive_ir_years": int((positive.information_ratio > 0).sum()), "mlp_outperforms_factor_mean_years": int((delta.delta_7_5bp_return > 0).sum()), "stitched_mlp_outperforms_factor_mean": bool(overall["Baseline MLP"]["return_7_5bp_annualized"] > overall["Factor Mean"]["return_7_5bp_annualized"]), "return_concentration_max_year_share": float(max(positive.return_7_5bp_annualized.clip(lower=0)) / max(positive.return_7_5bp_annualized.clip(lower=0).sum(), 1e-12)), "failed_years": positive.loc[positive.annual_excess_return <= 0, "year"].tolist(), "switch_turnover_max": float(switches.one_way_turnover.max())}
    report = ["# Fixed-64 Annual Walk-Forward Baseline", "", "## Fixed setup", "", "- Fixed frozen 64 factor names and directions; no fold re-selection or re-direction.", "- MLP: Linear(64,64) → LeakyReLU(0.2) → Linear(64,1).", f"- C/D objective: L_score + {COST_BPS and 70.2102429073} × L_gross/net; D uses 7.5bp one-side cost and lr=1e-4.", "- Allocator: cross-sectional z-score then masked softmax, T=1.0; no TopK/cap in C/D.", "", "## Fold table", "", _markdown(folds, ["fold", "train_start", "train_end", "validation_start", "validation_end", "oos_start", "oos_end", "a_best", "c_best", "d_stability_best", "full_train_rankic", "full_validation_rankic", "true_rankic_gap"]), "", "## OOS annual table", "", _markdown(annual, ["year", "strategy", "return_7_5bp_annualized", "sharpe_7_5bp", "mdd_7_5bp", "mean_one_way_turnover", "mean_effective_n", "annual_excess_return", "information_ratio"]), "", "## MLP minus Factor Mean", "", _markdown(delta, list(delta.columns)), "", "## Model-switch turnover", "", _markdown(switches, list(switches.columns)), "", "## Stability diagnosis", "", f"- Positive net-return years: {diagnosis['baseline_positive_net_years']}/5; positive-excess years: {diagnosis['baseline_positive_excess_years']}/5; IR>0 years: {diagnosis['baseline_positive_ir_years']}/5.", f"- MLP beats Factor Mean in {diagnosis['mlp_outperforms_factor_mean_years']}/5 annual net-return comparisons; stitched outperformance: {diagnosis['stitched_mlp_outperforms_factor_mean']}.", f"- Non-positive-excess years: {diagnosis['failed_years']}; maximum model-switch turnover: {diagnosis['switch_turnover_max']:.6f}.", "", "All formal annual OOS metrics above are sliced from the one continuous 2022–2026YTD stitched accounting path; per-fold OOS files are diagnostic only."]
    (destination / "stitched_oos_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    reports = root / "reports"; reports.mkdir(exist_ok=True)
    shutil_target = reports / "stitched_oos_report.md"
    shutil_target.write_text((destination / "stitched_oos_report.md").read_text(encoding="utf-8"), encoding="utf-8")
    root_manifest = {"status": "PASS", "completed_at_utc": datetime.now(UTC).isoformat(), "fixed_factor_hashes": json.loads((root / "data_preflight" / "fixed_factor_audit.json").read_text(encoding="utf-8")), "folds": [0, 1, 2, 3, 4], "factor_reselection_invoked": False, "temperature": 1.0, "lambda_portfolio": 70.2102429073, "network_modified": False, "portfolio_constraints_added": False, "n1_or_later_run": False, "annual_metrics_from_continuous_stitched_accounting": True, "model_switch_no_free_turnover": True, "factor_mean_included": True}
    _json(root / "run_manifest.json", root_manifest)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fixed64_annual_walk_forward_baseline.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/cloud/E2EAI/output/fixed64_annual_walk_forward_baseline"))
    args = parser.parse_args()
    print(stitch(args.config, args.output))


if __name__ == "__main__":
    main()
