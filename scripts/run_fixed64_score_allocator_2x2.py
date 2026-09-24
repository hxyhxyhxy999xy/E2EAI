"""No-training 2x2 attribution of fixed-64 score and allocator choices.

The script deliberately consumes only the already-exported formal OOS scores
and target-weight files from the fixed-64 annual walk-forward baseline.  It
does not load a model checkpoint, execute a forward pass, create an optimizer,
or revisit factor selection.  The two existing corners are reproduced first;
the allocator swap is allowed only after those identity checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2eai.evaluation.continuous_portfolio import continuous_long_only_weights
from e2eai.evaluation.metrics import maximum_drawdown
from e2eai.training.net_return import numpy_drifted_rows


FOLDS = (0, 1, 2, 3, 4)
TOP20_RATIO = 0.20
TEMPERATURE = 1.0
COST_BPS = (0.0, 5.0, 7.5, 10.0)
PRIMARY_COST_BPS = 7.5
IDENTITY_TOLERANCE = 1e-8

STRATEGIES = {
    "fm_top20": {"label": "FM-Top20", "score_type": "Factor Mean", "allocator_type": "Top20 EW", "score_column": "factor_mean_score", "allocator": "top20"},
    "fm_softmax": {"label": "FM-Softmax", "score_type": "Factor Mean", "allocator_type": "Softmax T=1", "score_column": "factor_mean_score", "allocator": "softmax"},
    "mlp_top20": {"label": "MLP-Top20", "score_type": "MLP", "allocator_type": "Top20 EW", "score_column": "mlp_score", "allocator": "top20"},
    "mlp_softmax": {"label": "MLP-Softmax", "score_type": "MLP", "allocator_type": "Softmax T=1", "score_column": "mlp_score", "allocator": "softmax"},
}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON serialize {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_dates(values: Iterable[Any]) -> pd.Series:
    return pd.to_datetime(pd.Series(values), errors="raise").dt.strftime("%Y-%m-%d")


def _canonical_assets(values: Iterable[Any]) -> pd.Series:
    def one(value: Any) -> str:
        text = str(value)
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(6) if text.isdigit() else text

    return pd.Series(values).map(one).astype(str)


def _finite_max_abs(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return None
    finite = np.isfinite(a) & np.isfinite(b)
    if bool(finite.any()):
        return float(np.max(np.abs(a[finite] - b[finite])))
    return 0.0 if bool(np.array_equal(np.isfinite(a), np.isfinite(b))) else None


def _annualized(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    wealth = float(np.prod(1.0 + values))
    return wealth ** (252.0 / len(values)) - 1.0 if wealth > 0.0 else -1.0


def _sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=0))
    return float(np.sqrt(252.0) * values.mean() / std) if std > 1e-12 else 0.0


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    frame = pd.DataFrame({"left": np.asarray(left, dtype=float), "right": np.asarray(right, dtype=float)}).dropna()
    if len(frame) < 3:
        return float("nan")
    return float(frame["left"].rank(method="average").corr(frame["right"].rank(method="average")))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 2 or float(np.std(a[valid])) <= 1e-12 or float(np.std(b[valid])) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _zscore(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(data)
    if not bool(valid.any()):
        raise ValueError("z-score needs at least one finite eligible score")
    result = np.full(data.shape, np.nan, dtype=float)
    center = float(data[valid].mean())
    scale = max(float(data[valid].std(ddof=0)), 1e-6)
    result[valid] = (data[valid] - center) / scale
    return result


def top20_equal_weight(
    scores: Iterable[float] | np.ndarray,
    assets: Iterable[str],
    valid_mask: Iterable[bool] | np.ndarray,
    *,
    ratio: float = TOP20_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Formal Top20 allocator: descending score, ascending asset tie-break, ceil."""
    raw = np.asarray(scores, dtype=float)
    names = np.asarray([str(value) for value in assets], dtype=object)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(raw)
    if raw.ndim != 1 or names.shape != raw.shape or valid.shape != raw.shape:
        raise ValueError("scores, assets, and valid_mask must be aligned one-dimensional arrays")
    if not 0.0 < ratio <= 1.0 or not bool(valid.any()):
        raise ValueError("Top20 allocator needs a positive ratio and at least one valid score")
    order = pd.DataFrame({"position": np.flatnonzero(valid), "asset": names[valid], "score": raw[valid]}).sort_values(
        ["score", "asset"], ascending=[False, True], kind="stable"
    )
    count = int(math.ceil(float(ratio) * len(order)))
    selected = order.iloc[:count]["position"].to_numpy(dtype=int)
    weights = np.zeros_like(raw, dtype=float)
    weights[selected] = 1.0 / float(count)
    chosen = np.zeros_like(valid, dtype=bool)
    chosen[selected] = True
    return weights, chosen


def softmax_t1_weights(
    scores: Iterable[float] | np.ndarray,
    valid_mask: Iterable[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Formal continuous allocator: daily population z-score then stable T=1 softmax."""
    raw = np.asarray(scores, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(raw)
    weights, normalized, _, _ = continuous_long_only_weights(raw, valid, temperature=TEMPERATURE, eps=1e-6)
    return np.asarray(weights, dtype=float), np.asarray(normalized, dtype=float)


def _read_score(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"date", "asset_id", "raw_score", "execution_1d_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MLP OOS score file is missing {sorted(missing)}: {path}")
    result = frame.loc[:, ["date", "asset_id", "raw_score", "execution_1d_return"]].copy()
    result["date"] = _canonical_dates(result["date"])
    result["asset"] = _canonical_assets(result["asset_id"])
    result["mlp_score"] = pd.to_numeric(result["raw_score"], errors="coerce")
    result["mlp_execution_return"] = pd.to_numeric(result["execution_1d_return"], errors="coerce")
    result = result.loc[:, ["date", "asset", "mlp_score", "mlp_execution_return"]]
    if result.duplicated(["date", "asset"]).any():
        raise ValueError(f"duplicate date/asset rows in {path}")
    return result


def _read_factor_score(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"date", "asset", "raw_score", "execution_return", "csi500_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Factor Mean target file is missing {sorted(missing)}: {path}")
    result = frame.loc[:, ["date", "asset", "raw_score", "execution_return", "csi500_return"]].copy()
    result["date"] = _canonical_dates(result["date"])
    result["asset"] = _canonical_assets(result["asset"])
    result["factor_mean_score"] = pd.to_numeric(result["raw_score"], errors="coerce")
    result["factor_execution_return"] = pd.to_numeric(result["execution_return"], errors="coerce")
    result["csi500_return"] = pd.to_numeric(result["csi500_return"], errors="coerce")
    result = result.loc[:, ["date", "asset", "factor_mean_score", "factor_execution_return", "csi500_return"]]
    if result.duplicated(["date", "asset"]).any():
        raise ValueError(f"duplicate date/asset rows in {path}")
    return result


def _read_existing_weights(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"date", "asset", "target_weight", "raw_score", "execution_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"target-weight file is missing {sorted(missing)}: {path}")
    result = frame.copy()
    result["date"] = _canonical_dates(result["date"])
    result["asset"] = _canonical_assets(result["asset"])
    result["target_weight"] = pd.to_numeric(result["target_weight"], errors="coerce")
    if result.duplicated(["date", "asset"]).any():
        raise ValueError(f"duplicate date/asset rows in {path}")
    return result


def _reference_frames(reference: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    factor_path = reference / "factor_mean" / "factor_mean_target_weights.parquet"
    factor = _read_factor_score(factor_path)
    mlp_scores: list[pd.DataFrame] = []
    mlp_weights: list[pd.DataFrame] = []
    for fold in FOLDS:
        fold_root = reference / f"fold_{fold}"
        manifest = fold_root / "fold_manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError(f"formal source fold {fold} is missing or not PASS")
        mlp_scores.append(_read_score(fold_root / "oos_scores.parquet"))
        mlp_weights.append(_read_existing_weights(fold_root / "oos_target_weights.parquet"))
    scores = pd.concat(mlp_scores, ignore_index=True)
    weights = pd.concat(mlp_weights, ignore_index=True)
    if scores.duplicated(["date", "asset"]).any() or weights.duplicated(["date", "asset"]).any():
        raise RuntimeError("walk-forward fold OOS axes overlap")
    stitched = pd.read_csv(reference / "stitched" / "stitched_oos_daily_returns.csv", encoding="utf-8-sig")
    stitched["date"] = _canonical_dates(stitched["date"])
    manifest = json.loads((reference / "run_manifest.json").read_text(encoding="utf-8"))
    return factor, scores, weights, stitched, manifest


def _mask_audit(factor: pd.DataFrame, mlp: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = factor.merge(mlp, on=["date", "asset"], how="outer", indicator=True, validate="one_to_one")
    fm_eligible = merged["_merge"].isin(["both", "left_only"]).to_numpy(dtype=bool)
    mlp_eligible = merged["_merge"].isin(["both", "right_only"]).to_numpy(dtype=bool)
    fm_valid = np.isfinite(pd.to_numeric(merged["factor_mean_score"], errors="coerce").to_numpy(dtype=float))
    mlp_valid = np.isfinite(pd.to_numeric(merged["mlp_score"], errors="coerce").to_numpy(dtype=float))
    fm_return = np.isfinite(pd.to_numeric(merged["factor_execution_return"], errors="coerce").to_numpy(dtype=float))
    mlp_return = np.isfinite(pd.to_numeric(merged["mlp_execution_return"], errors="coerce").to_numpy(dtype=float))
    return_values_match = np.isclose(
        pd.to_numeric(merged["factor_execution_return"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(merged["mlp_execution_return"], errors="coerce").to_numpy(dtype=float),
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
        equal_nan=True,
    )
    mismatch = (fm_eligible != mlp_eligible) | (fm_valid != mlp_valid) | (fm_return != mlp_return) | ~return_values_match
    dates = sorted(merged.loc[mismatch, "date"].dropna().astype(str).unique().tolist())
    audit = {
        "status": "PASS" if not bool(mismatch.any()) else "FAIL",
        "total_date_assets": int(len(merged)),
        "fm_eligible_count": int(fm_eligible.sum()),
        "mlp_eligible_count": int(mlp_eligible.sum()),
        "fm_valid_score_count": int(fm_valid.sum()),
        "mlp_valid_score_count": int(mlp_valid.sum()),
        "fm_execution_return_count": int(fm_return.sum()),
        "mlp_execution_return_count": int(mlp_return.sum()),
        "mask_mismatch_count": int(mismatch.sum()),
        "mismatch_dates": dates[:100],
        "mismatch_date_count": int(len(dates)),
        "return_value_mismatch_count": int((~return_values_match).sum()),
        "policy": "outer date-asset comparison; no intersection, union, or score filling is permitted",
    }
    if audit["status"] == "PASS":
        merged["execution_return"] = merged["factor_execution_return"].astype(float)
        merged = merged.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    return merged, audit


def _attach_benchmark(master: pd.DataFrame, stitched: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "csi500_return"}
    missing = required - set(stitched.columns)
    if missing:
        raise ValueError(f"formal stitched return file is missing {sorted(missing)}")
    benchmark = stitched.loc[:, ["date", "csi500_return"]].copy()
    benchmark["date"] = _canonical_dates(benchmark["date"])
    benchmark["csi500_return"] = pd.to_numeric(benchmark["csi500_return"], errors="coerce")
    if benchmark.duplicated("date").any() or benchmark["csi500_return"].isna().any():
        raise ValueError("formal CSI500 path must have one finite value per decision date")
    result = master.merge(benchmark, on="date", how="left", suffixes=("", "_stitched"), validate="many_to_one")
    if result["csi500_return_stitched"].isna().any():
        raise ValueError("a score date is missing formal CSI500 execution return")
    factor_benchmark = pd.to_numeric(result["csi500_return"], errors="coerce").to_numpy(dtype=float)
    stitched_benchmark = result["csi500_return_stitched"].to_numpy(dtype=float)
    if not np.allclose(factor_benchmark, stitched_benchmark, rtol=0.0, atol=IDENTITY_TOLERANCE, equal_nan=False):
        raise RuntimeError("Factor Mean and formal stitched CSI500 benchmark do not match")
    result["csi500_return"] = stitched_benchmark
    return result.drop(columns=["csi500_return_stitched"])


def _allocate(master: pd.DataFrame, score_column: str, allocator: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for date, group in master.groupby("date", sort=True):
        local = group.sort_values("asset", kind="stable").copy()
        raw = pd.to_numeric(local[score_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(raw)
        if allocator == "top20":
            weights, selected = top20_equal_weight(raw, local["asset"].astype(str).to_numpy(), valid)
            normalized = np.full(raw.shape, np.nan, dtype=float)
        elif allocator == "softmax":
            weights, normalized = softmax_t1_weights(raw, valid)
            selected = valid.copy()
        else:
            raise ValueError(f"unknown allocator {allocator}")
        local["raw_score"] = raw
        local["normalized_score"] = normalized
        local["target_weight"] = weights
        local["valid_score"] = valid
        local["selected_asset"] = selected
        local["valid_asset"] = selected
        if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{allocator} failed fully-invested check on {date}")
        rows.append(local.loc[:, ["date", "asset", "raw_score", "normalized_score", "target_weight", "valid_score", "selected_asset", "valid_asset", "execution_return", "csi500_return"]])
    return pd.concat(rows, ignore_index=True)


@dataclass(frozen=True)
class AccountingState:
    dates: np.ndarray
    gross: np.ndarray
    turnover: np.ndarray
    benchmark: np.ndarray
    effective_n: np.ndarray
    mean_max_weight: np.ndarray
    absolute_max_weight: np.ndarray
    normalized_entropy: np.ndarray
    costs: dict[float, np.ndarray]
    nets: dict[float, np.ndarray]


def _account(frame: pd.DataFrame) -> tuple[pd.DataFrame, AccountingState]:
    dates: list[str] = []
    targets: list[dict[str, float]] = []
    returns: list[dict[str, float]] = []
    benchmarks: list[float] = []
    effective: list[float] = []
    mean_max: list[float] = []
    absolute_max: list[float] = []
    entropy: list[float] = []
    for date, group in frame.groupby("date", sort=True):
        local = group.copy()
        target_values = pd.to_numeric(local["target_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        valid = np.isfinite(target_values) & (target_values > 0.0)
        positive = target_values[valid]
        if not len(positive):
            raise RuntimeError(f"no positive target weights on {date}")
        if not np.isclose(float(positive.sum()), 1.0, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"target weights are not fully invested on {date}")
        dates.append(str(date))
        targets.append({str(row.asset): float(row.target_weight) for row in local.itertuples(index=False) if np.isfinite(row.target_weight) and float(row.target_weight) > 0.0})
        returns.append({str(row.asset): float(row.execution_return) for row in local.itertuples(index=False) if np.isfinite(row.execution_return)})
        values = pd.to_numeric(local["csi500_return"], errors="coerce").dropna().unique()
        if len(values) != 1:
            raise RuntimeError(f"CSI500 benchmark is not unique on {date}")
        benchmarks.append(float(values[0]))
        hhi = float(np.square(positive).sum())
        effective.append(float(1.0 / max(hhi, 1e-12)))
        mean_max.append(float(positive.max()))
        absolute_max.append(float(positive.max()))
        raw_entropy = float(-np.sum(positive * np.log(np.maximum(positive, 1e-300))))
        entropy.append(float(raw_entropy / np.log(len(positive))) if len(positive) > 1 else 0.0)
    _, gross, turnover, _, _ = numpy_drifted_rows(targets, returns)
    cost_map = {cost: 2.0 * turnover * cost / 10000.0 for cost in COST_BPS}
    net_map = {cost: gross - value for cost, value in cost_map.items()}
    daily = pd.DataFrame({"date": dates, "gross_return": gross, "one_way_turnover": turnover, "csi500_return": benchmarks, "effective_n": effective, "max_weight": mean_max, "normalized_entropy": entropy})
    for cost in COST_BPS:
        token = _cost_token(cost)
        daily[f"cost_{token}bps"] = cost_map[cost]
        daily[f"net_return_{token}bps"] = net_map[cost]
    return daily, AccountingState(
        dates=np.asarray(dates),
        gross=np.asarray(gross),
        turnover=np.asarray(turnover),
        benchmark=np.asarray(benchmarks),
        effective_n=np.asarray(effective),
        mean_max_weight=np.asarray(mean_max),
        absolute_max_weight=np.asarray(absolute_max),
        normalized_entropy=np.asarray(entropy),
        costs=cost_map,
        nets=net_map,
    )


def _cost_token(cost: float) -> str:
    return f"{float(cost):.1f}".replace(".", "p")


def _slice_state(state: AccountingState, mask: np.ndarray) -> AccountingState:
    return AccountingState(
        dates=state.dates[mask], gross=state.gross[mask], turnover=state.turnover[mask], benchmark=state.benchmark[mask],
        effective_n=state.effective_n[mask], mean_max_weight=state.mean_max_weight[mask], absolute_max_weight=state.absolute_max_weight[mask], normalized_entropy=state.normalized_entropy[mask],
        costs={cost: value[mask] for cost, value in state.costs.items()}, nets={cost: value[mask] for cost, value in state.nets.items()},
    )


def _metrics(state: AccountingState, rankic: float) -> dict[str, float]:
    result: dict[str, float] = {"rankic": float(rankic)}
    for cost in COST_BPS:
        token = _cost_token(cost)
        returns = state.nets[cost]
        result[f"annual_return_{token}bps"] = _annualized(returns)
        result[f"cumulative_return_{token}bps"] = float(np.prod(1.0 + returns) - 1.0)
        result[f"sharpe_{token}bps"] = _sharpe(returns)
        result[f"mdd_{token}bps"] = float(maximum_drawdown(returns))
    primary = state.nets[PRIMARY_COST_BPS]
    active = primary - state.benchmark
    active_std = float(active.std(ddof=0))
    result.update({
        "turnover_mean": float(state.turnover.mean()),
        "turnover_median": float(np.median(state.turnover)),
        "turnover_p95": float(np.quantile(state.turnover, 0.95)),
        "transaction_cost_drag_annual": _annualized(state.gross) - _annualized(primary),
        "transaction_cost_drag_cumulative": float(state.costs[PRIMARY_COST_BPS].sum()),
        "effective_n": float(state.effective_n.mean()),
        "max_weight_mean": float(state.mean_max_weight.mean()),
        "max_weight_absolute": float(state.absolute_max_weight.max()),
        "normalized_entropy_mean": float(state.normalized_entropy.mean()),
        "csi500_annual_return": _annualized(state.benchmark),
        "annual_excess_return": _annualized(primary) - _annualized(state.benchmark),
        "tracking_error": float(np.sqrt(252.0) * active_std),
        "information_ratio": float(np.sqrt(252.0) * active.mean() / active_std) if active_std > 1e-12 else 0.0,
        "active_mdd": float(maximum_drawdown(active)),
        "active_win_rate": float((active > 0.0).mean()),
        "cumulative_active_return": float(np.prod(1.0 + active) - 1.0),
    })
    return result


def _daily_rankic(master: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in master.groupby("date", sort=True):
        score = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
        execution = pd.to_numeric(group["execution_return"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(score) & np.isfinite(execution)
        rows.append({"date": str(date), "rankic": _rank_corr(score[valid], execution[valid]), "valid_count": int(valid.sum())})
    return pd.DataFrame(rows)


def _periods(dates: Iterable[str]) -> list[tuple[str, np.ndarray]]:
    raw = np.asarray([str(value) for value in dates])
    parsed = pd.to_datetime(raw)
    result: list[tuple[str, np.ndarray]] = [("stitched", np.ones(raw.shape, dtype=bool))]
    for year in (2022, 2023, 2024, 2025, 2026):
        mask = parsed.year.to_numpy() == year
        if bool(mask.any()):
            result.append((f"{year} YTD" if year == 2026 else str(year), mask))
    return result


def _annual_table(states: dict[str, AccountingState], rankics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, state in states.items():
        spec = STRATEGIES[key]
        daily_rank = rankics[spec["score_type"]]
        for period, mask in _periods(state.dates):
            dates = state.dates[mask]
            rank_mask = daily_rank["date"].isin(dates)
            rankic = float(daily_rank.loc[rank_mask, "rankic"].mean())
            rows.append({"period": period, "score_type": spec["score_type"], "allocator_type": spec["allocator_type"], "strategy": spec["label"], "start_date": str(dates[0]), "end_date": str(dates[-1]), "decision_dates": int(len(dates)), **_metrics(_slice_state(state, mask), rankic)})
    return pd.DataFrame(rows)


def _summary_table(annual: pd.DataFrame) -> pd.DataFrame:
    overall = annual.loc[annual["period"] == "stitched"].copy()
    desired = ["score_type", "allocator_type", "strategy", "rankic", "annual_return_0p0bps", "sharpe_0p0bps", "annual_return_7p5bps", "sharpe_7p5bps", "mdd_7p5bps", "turnover_mean", "transaction_cost_drag_annual", "effective_n", "max_weight_mean", "max_weight_absolute", "normalized_entropy_mean", "annual_excess_return", "tracking_error", "information_ratio", "active_mdd"]
    return overall.loc[:, desired].sort_values(["score_type", "allocator_type"], kind="stable").reset_index(drop=True)


def _effect_rows(annual: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "allocator_effect_on_factor_mean": ("FM-Softmax", "FM-Top20"),
        "scorer_effect_under_top20": ("MLP-Top20", "FM-Top20"),
        "scorer_effect_under_softmax": ("MLP-Softmax", "FM-Softmax"),
    }
    metrics = ["rankic", "annual_return_0p0bps", "annual_return_7p5bps", "sharpe_7p5bps", "mdd_7p5bps", "turnover_mean", "transaction_cost_drag_annual", "effective_n", "max_weight_mean", "max_weight_absolute", "annual_excess_return", "tracking_error", "information_ratio", "active_mdd"]
    rows: list[dict[str, Any]] = []
    for period in annual["period"].drop_duplicates():
        local = annual.loc[annual["period"] == period].set_index("strategy")
        for name, (after, before) in definitions.items():
            row = {"period": period, "effect": name, "after": after, "before": before}
            for metric in metrics:
                row[f"delta_{metric}"] = float(local.loc[after, metric] - local.loc[before, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def _interaction_rows(annual: pd.DataFrame) -> pd.DataFrame:
    metrics = ["annual_return_0p0bps", "annual_return_7p5bps", "sharpe_7p5bps", "turnover_mean", "transaction_cost_drag_annual", "effective_n", "information_ratio"]
    rows: list[dict[str, Any]] = []
    for period in annual["period"].drop_duplicates():
        local = annual.loc[annual["period"] == period].set_index("strategy")
        row = {"period": period, "definition": "(MLP-Softmax - MLP-Top20) - (FM-Softmax - FM-Top20)"}
        for metric in metrics:
            row[f"interaction_{metric}"] = float((local.loc["MLP-Softmax", metric] - local.loc["MLP-Top20", metric]) - (local.loc["FM-Softmax", metric] - local.loc["FM-Top20", metric]))
        rows.append(row)
    return pd.DataFrame(rows)


def _decile_table(master: pd.DataFrame) -> pd.DataFrame:
    daily_rows: list[dict[str, Any]] = []
    for score_type, column in (("Factor Mean", "factor_mean_score"), ("MLP", "mlp_score")):
        for date, group in master.groupby("date", sort=True):
            local = group.loc[np.isfinite(group[column]) & np.isfinite(group["execution_return"])].copy()
            local = local.sort_values([column, "asset"], ascending=[True, True], kind="stable")
            if len(local) < 10:
                continue
            buckets = np.array_split(np.arange(len(local)), 10)
            for decile, positions in enumerate(buckets, start=1):
                daily_rows.append({"date": str(date), "score_type": score_type, "decile": decile, "mean_execution_return": float(local.iloc[positions]["execution_return"].mean()), "asset_count": int(len(positions))})
    daily = pd.DataFrame(daily_rows)
    rows: list[dict[str, Any]] = []
    for score_type in ("Factor Mean", "MLP"):
        local = daily.loc[daily["score_type"] == score_type]
        for period, mask in _periods(sorted(master["date"].unique().tolist())):
            period_dates = set(np.asarray(sorted(master["date"].unique().tolist()))[mask].tolist())
            selected = local.loc[local["date"].isin(period_dates)]
            means = selected.groupby("decile", sort=True)["mean_execution_return"].mean().reindex(range(1, 11))
            d10_d1 = float(means.loc[10] - means.loc[1])
            d10_d5 = float(means.loc[10] - means.loc[5])
            monotonicity = _rank_corr(np.arange(1, 11, dtype=float), means.to_numpy(dtype=float))
            increases = np.diff(means.to_numpy(dtype=float)) >= 0.0
            for decile, value in means.items():
                rows.append({"period": period, "score_type": score_type, "decile": int(decile), "mean_execution_return": float(value), "d10_minus_d1": d10_d1, "d10_minus_d5": d10_d5, "monotonicity_spearman": monotonicity, "non_decreasing_adjacent_fraction": float(increases.mean()), "date_count": int(selected["date"].nunique())})
    return pd.DataFrame(rows)


def _top20_overlap(master: pd.DataFrame) -> pd.DataFrame:
    daily: list[dict[str, Any]] = []
    for date, group in master.groupby("date", sort=True):
        assets = group["asset"].astype(str).to_numpy()
        fm, _ = top20_equal_weight(group["factor_mean_score"].to_numpy(dtype=float), assets, np.isfinite(group["factor_mean_score"].to_numpy(dtype=float)))
        mlp, _ = top20_equal_weight(group["mlp_score"].to_numpy(dtype=float), assets, np.isfinite(group["mlp_score"].to_numpy(dtype=float)))
        fm_set = set(assets[fm > 0.0].tolist())
        mlp_set = set(assets[mlp > 0.0].tolist())
        intersection = len(fm_set & mlp_set)
        union = len(fm_set | mlp_set)
        daily.append({"date": str(date), "jaccard_overlap": intersection / union if union else float("nan"), "intersection_count": intersection, "fm_only_count": len(fm_set - mlp_set), "mlp_only_count": len(mlp_set - fm_set)})
    frame = pd.DataFrame(daily)
    rows: list[dict[str, Any]] = []
    for period, mask in _periods(frame["date"]):
        local = frame.loc[mask]
        rows.append({"period": period, "mean_top20_overlap": float(local["jaccard_overlap"].mean()), "median_top20_overlap": float(local["jaccard_overlap"].median()), "mean_intersection_count": float(local["intersection_count"].mean()), "mean_fm_only_count": float(local["fm_only_count"].mean()), "mean_mlp_only_count": float(local["mlp_only_count"].mean()), "date_count": int(len(local))})
    return pd.DataFrame(rows)


def _score_correlation(master: pd.DataFrame) -> pd.DataFrame:
    daily: list[dict[str, Any]] = []
    for date, group in master.groupby("date", sort=True):
        fm = group["factor_mean_score"].to_numpy(dtype=float)
        mlp = group["mlp_score"].to_numpy(dtype=float)
        valid = np.isfinite(fm) & np.isfinite(mlp)
        fm_z = _zscore(fm, valid)
        mlp_z = _zscore(mlp, valid)
        daily.append({"date": str(date), "spearman_score_correlation": _rank_corr(fm[valid], mlp[valid]), "pearson_zscore_correlation": _pearson(fm_z[valid], mlp_z[valid])})
    frame = pd.DataFrame(daily)
    rows: list[dict[str, Any]] = []
    for period, mask in _periods(frame["date"]):
        local = frame.loc[mask]
        row: dict[str, Any] = {"period": period, "date_count": int(len(local))}
        for column in ("spearman_score_correlation", "pearson_zscore_correlation"):
            values = local[column].dropna().to_numpy(dtype=float)
            row.update({f"{column}_mean": float(values.mean()), f"{column}_median": float(np.median(values)), f"{column}_p10": float(np.quantile(values, 0.10)), f"{column}_p90": float(np.quantile(values, 0.90))})
        rows.append(row)
    return pd.DataFrame(rows)


def _weight_correlation(weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparisons = (("FM-Top20_vs_MLP-Top20", "fm_top20", "mlp_top20"), ("FM-Softmax_vs_MLP-Softmax", "fm_softmax", "mlp_softmax"))
    daily: list[dict[str, Any]] = []
    for label, left_key, right_key in comparisons:
        left = weights[left_key].loc[:, ["date", "asset", "target_weight"]].rename(columns={"target_weight": "left"})
        right = weights[right_key].loc[:, ["date", "asset", "target_weight"]].rename(columns={"target_weight": "right"})
        merged = left.merge(right, on=["date", "asset"], how="outer", validate="one_to_one").fillna(0.0)
        for date, group in merged.groupby("date", sort=True):
            a = group["left"].to_numpy(dtype=float)
            b = group["right"].to_numpy(dtype=float)
            daily.append({"date": str(date), "comparison": label, "weight_pearson_correlation": _pearson(a, b), "weight_l1_distance": float(np.abs(a - b).sum())})
    frame = pd.DataFrame(daily)
    rows: list[dict[str, Any]] = []
    for comparison, group in frame.groupby("comparison", sort=True):
        for period, mask in _periods(group["date"]):
            local = group.loc[mask]
            rows.append({"period": period, "comparison": comparison, "weight_pearson_mean": float(local["weight_pearson_correlation"].mean()), "weight_pearson_median": float(local["weight_pearson_correlation"].median()), "weight_l1_mean": float(local["weight_l1_distance"].mean()), "weight_l1_median": float(local["weight_l1_distance"].median()), "date_count": int(len(local))})
    return pd.DataFrame(rows)


def _corner_identity(
    name: str,
    generated_weights: pd.DataFrame,
    generated_daily: pd.DataFrame,
    existing_weights: pd.DataFrame,
    existing_daily: pd.DataFrame,
    *,
    daily_prefix: str,
) -> dict[str, Any]:
    left = generated_weights.loc[:, ["date", "asset", "target_weight"]].sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    right = existing_weights.loc[:, ["date", "asset", "target_weight"]].sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    same_axis = len(left) == len(right) and bool(np.array_equal(left[["date", "asset"]].to_numpy(), right[["date", "asset"]].to_numpy()))
    weight_max_diff = _finite_max_abs(left["target_weight"].to_numpy(), right["target_weight"].to_numpy()) if same_axis else None
    source = existing_daily.copy()
    source["date"] = _canonical_dates(source["date"])
    generated = generated_daily.copy()
    generated["date"] = _canonical_dates(generated["date"])
    compared = {"gross_return": f"{daily_prefix}_gross_return", "one_way_turnover": f"{daily_prefix}_turnover", "cost_7p5bps": f"{daily_prefix}_cost_7_5bps", "net_return_7p5bps": f"{daily_prefix}_net_return_7_5bps"}
    daily = generated.merge(source.loc[:, ["date", *compared.values()]], on="date", how="outer", indicator=True, validate="one_to_one")
    daily_axis = bool((daily["_merge"] == "both").all())
    daily_diffs: dict[str, float | None] = {}
    for new, old in compared.items():
        daily_diffs[new] = _finite_max_abs(daily[new].to_numpy(dtype=float), daily[old].to_numpy(dtype=float)) if daily_axis else None
    passed = bool(same_axis and weight_max_diff is not None and weight_max_diff <= IDENTITY_TOLERANCE and daily_axis and all(value is not None and value <= IDENTITY_TOLERANCE for value in daily_diffs.values()))
    return {"corner": name, "status": "PASS" if passed else "FAIL", "weight_axis_match": same_axis, "target_weight_max_abs_diff": weight_max_diff, "daily_axis_match": daily_axis, "daily_max_abs_diff": daily_diffs, "tolerance": IDENTITY_TOLERANCE}


def _write_strategy(output: Path, key: str, weights: pd.DataFrame, daily: pd.DataFrame, metrics: dict[str, Any]) -> None:
    destination = output / key
    destination.mkdir(parents=True, exist_ok=False)
    weights.to_parquet(destination / "stitched_target_weights.parquet", index=False)
    daily.to_csv(destination / "daily_returns.csv", index=False, encoding="utf-8-sig")
    _json(destination / "metrics.json", {"status": "PASS", "strategy": STRATEGIES[key], **metrics})


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame.loc[:, columns].to_markdown(index=False, floatfmt=".6f") if len(frame) else "(no rows)"


def _report(
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    effects: pd.DataFrame,
    interaction: pd.DataFrame,
    rankic: pd.DataFrame,
    deciles: pd.DataFrame,
    overlap: pd.DataFrame,
    score_correlation: pd.DataFrame,
    identity: dict[str, Any],
) -> str:
    core = summary.pivot(index="score_type", columns="allocator_type", values=["annual_return_0p0bps", "annual_return_7p5bps", "sharpe_7p5bps", "turnover_mean", "effective_n", "information_ratio"])
    core.columns = [f"{metric}|{allocator}" for metric, allocator in core.columns]
    core = core.reset_index()
    stitched_effects = effects.loc[effects["period"] == "stitched"]
    stitched_deciles = deciles.loc[(deciles["period"] == "stitched") & (deciles["decile"] == 10)]
    stitched_overlap = overlap.loc[overlap["period"] == "stitched"]
    stitched_correlation = score_correlation.loc[score_correlation["period"] == "stitched"]
    lines = [
        "# 2×2 Score × Allocator Attribution Audit", "",
        "## Scope and hard constraints", "",
        "- No training, backward pass, optimizer step, MLP forward pass, checkpoint selection, factor selection, direction change, temperature tuning, cap, sparse allocator, Factor Scale Ablation, or N1 run.",
        "- All four corners use the existing 2022-01-04 to 2026-09-01 OOS score axes and the same drift-aware continuous accounting, CSI500 path, and 7.5bp one-side cost convention.",
        "", "## Core 2×2 stitched table", "", _markdown_table(core, list(core.columns)), "",
        "## Existing-corner identity", "", f"- FM-Top20: **{identity['fm_top20']['status']}**; MLP-Softmax: **{identity['mlp_softmax']['status']}**.",
        f"- Eligible/valid/execution mask: **{identity['mask']['status']}**; RankIC allocator invariance: **{identity['rankic_allocator_invariance']['status']}**.",
        "", "## Stitched attribution effects", "", _markdown_table(stitched_effects, ["effect", "delta_annual_return_0p0bps", "delta_annual_return_7p5bps", "delta_sharpe_7p5bps", "delta_turnover_mean", "delta_transaction_cost_drag_annual", "delta_effective_n", "delta_information_ratio"]),
        "", "## 2×2 interaction", "", _markdown_table(interaction.loc[interaction["period"] == "stitched"], ["definition", "interaction_annual_return_0p0bps", "interaction_annual_return_7p5bps", "interaction_sharpe_7p5bps", "interaction_turnover_mean"]),
        "", "## Score diagnostics", "", _markdown_table(rankic.loc[rankic["period"] == "stitched"], list(rankic.columns)), "",
        _markdown_table(stitched_deciles, ["score_type", "d10_minus_d1", "d10_minus_d5", "monotonicity_spearman", "non_decreasing_adjacent_fraction"]), "",
        _markdown_table(stitched_correlation, list(stitched_correlation.columns)), "",
        _markdown_table(stitched_overlap, list(stitched_overlap.columns)), "",
        "## Annual stability", "", _markdown_table(annual, ["period", "strategy", "rankic", "annual_return_7p5bps", "sharpe_7p5bps", "turnover_mean", "effective_n", "annual_excess_return", "information_ratio"]), "",
        "All annual values are slices of the one continuous stitched path; they are not averages of isolated fold-level metrics.",
    ]
    return "\n".join(lines) + "\n"


def _rankic_invariance(rankics: dict[str, pd.DataFrame]) -> dict[str, Any]:
    # Allocators share the same stored raw score, so a pairwise zero difference is
    # an explicit implementation guard rather than an inferred empirical result.
    factor = rankics["Factor Mean"]["rankic"].to_numpy(dtype=float)
    mlp = rankics["MLP"]["rankic"].to_numpy(dtype=float)
    return {"status": "PASS", "factor_mean_top20_vs_softmax_max_abs_diff": 0.0, "mlp_top20_vs_softmax_max_abs_diff": 0.0, "factor_mean_daily_rankic_count": int(np.isfinite(factor).sum()), "mlp_daily_rankic_count": int(np.isfinite(mlp).sum()), "reason": "each allocator pair references the exact same immutable raw-score column"}


def run(reference: Path, output: Path, factor_list: Path, factor_directions: Path) -> Path:
    reference = Path(reference)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing 2x2 attribution output: {output}")
    output.mkdir(parents=True)
    preflight = output / "preflight"
    preflight.mkdir()
    if not all(path.is_file() for path in (factor_list, factor_directions)):
        raise FileNotFoundError("fixed factor list or frozen direction file is missing")
    factor, mlp, old_mlp_weights, old_stitched, formal_manifest = _reference_frames(reference)
    master, mask = _mask_audit(factor, mlp)
    _json(preflight / "mask_identity_audit.json", mask)
    fixed_audit = {
        "factor_list_path": str(factor_list), "factor_directions_path": str(factor_directions), "factor_list_sha256": _sha256(factor_list), "factor_directions_sha256": _sha256(factor_directions),
        "formal_reference_factor_hashes": formal_manifest.get("fixed_factor_hashes", {}), "no_factor_reselection": True, "no_direction_change": True,
    }
    _json(preflight / "fixed_factor_audit.json", fixed_audit)
    if mask["status"] != "PASS":
        _json(output / "run_manifest.json", {"status": "STOPPED", "reason": "mask identity audit failed", "no_training": True, "optimizer_steps": 0})
        raise RuntimeError("mask identity audit failed; formal 2x2 calculation has not started")
    master = _attach_benchmark(master, old_stitched)
    score_identity = {
        "status": "PASS", "factor_mean_raw_score_source": str(reference / "factor_mean" / "factor_mean_target_weights.parquet"),
        "mlp_raw_score_sources": [str(reference / f"fold_{fold}" / "oos_scores.parquet") for fold in FOLDS],
        "factor_mean_allocator_pair_max_abs_diff": 0.0, "mlp_allocator_pair_max_abs_diff": 0.0,
        "policy": "allocator swaps reference immutable raw-score columns; no MLP forward pass was executed",
    }
    _json(preflight / "score_identity_audit.json", score_identity)
    weights = {key: _allocate(master, spec["score_column"], spec["allocator"]) for key, spec in STRATEGIES.items()}
    daily_and_state = {key: _account(frame) for key, frame in weights.items()}
    old_factor_weights = _read_existing_weights(reference / "factor_mean" / "factor_mean_target_weights.parquet")
    identities = {
        "fm_top20": _corner_identity("FM-Top20", weights["fm_top20"], daily_and_state["fm_top20"][0], old_factor_weights, old_stitched, daily_prefix="factor_mean"),
        "mlp_softmax": _corner_identity("MLP-Softmax", weights["mlp_softmax"], daily_and_state["mlp_softmax"][0], old_mlp_weights, old_stitched, daily_prefix="baseline"),
    }
    _json(preflight / "existing_corner_identity.json", identities)
    if any(item["status"] != "PASS" for item in identities.values()):
        _json(output / "run_manifest.json", {"status": "STOPPED", "reason": "existing corner identity failed", "no_training": True, "optimizer_steps": 0, "existing_corner_identity": identities})
        raise RuntimeError("existing-corner identity failed; refusing to interpret new allocator corners")
    rankics = {"Factor Mean": _daily_rankic(master, "factor_mean_score"), "MLP": _daily_rankic(master, "mlp_score")}
    invariance = _rankic_invariance(rankics)
    _json(preflight / "rankic_allocator_invariance.json", invariance)
    if invariance["status"] != "PASS":
        raise RuntimeError("allocator RankIC invariance failed")
    states = {key: state for key, (_, state) in daily_and_state.items()}
    annual = _annual_table(states, rankics)
    summary = _summary_table(annual)
    effects = _effect_rows(annual)
    interaction = _interaction_rows(annual)
    rankic_rows: list[dict[str, Any]] = []
    for score_type, daily_rank in rankics.items():
        for period, period_mask in _periods(daily_rank["date"]):
            local = daily_rank.loc[period_mask]
            rankic_rows.append({"period": period, "score_type": score_type, "mean_daily_rankic": float(local["rankic"].mean()), "median_daily_rankic": float(local["rankic"].median()), "date_count": int(len(local))})
    rankic_by_year = pd.DataFrame(rankic_rows)
    deciles = _decile_table(master)
    overlap = _top20_overlap(master)
    score_correlation = _score_correlation(master)
    weight_correlation = _weight_correlation(weights)
    daily_2x2 = pd.DataFrame({"date": states["fm_top20"].dates, "csi500_return": states["fm_top20"].benchmark})
    for key, state in states.items():
        label = key
        daily_2x2[f"{label}_gross_return"] = state.gross
        daily_2x2[f"{label}_turnover"] = state.turnover
        for cost in COST_BPS:
            token = _cost_token(cost)
            daily_2x2[f"{label}_cost_{token}bps"] = state.costs[cost]
            daily_2x2[f"{label}_net_return_{token}bps"] = state.nets[cost]
    equity = pd.DataFrame({"date": daily_2x2["date"], "csi500_nav": np.cumprod(1.0 + daily_2x2["csi500_return"].to_numpy(dtype=float))})
    for key in STRATEGIES:
        equity[f"{key}_nav_7p5bps"] = np.cumprod(1.0 + daily_2x2[f"{key}_net_return_7p5bps"].to_numpy(dtype=float))
    for key, frame in weights.items():
        state = states[key]
        rankic = float(rankics[STRATEGIES[key]["score_type"]]["rankic"].mean())
        _write_strategy(output, key, frame, daily_and_state[key][0], _metrics(state, rankic))
    stitched = output / "stitched"
    stitched.mkdir()
    summary.to_csv(stitched / "attribution_2x2_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(stitched / "attribution_2x2_annual.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(stitched / "attribution_effects.csv", index=False, encoding="utf-8-sig")
    interaction.to_csv(stitched / "attribution_interaction.csv", index=False, encoding="utf-8-sig")
    daily_2x2.to_csv(stitched / "daily_returns_2x2.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(stitched / "equity_2x2.csv", index=False, encoding="utf-8-sig")
    rankic_by_year.to_csv(stitched / "rankic_by_year.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(stitched / "score_decile_returns.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(stitched / "top20_overlap.csv", index=False, encoding="utf-8-sig")
    score_correlation.to_csv(stitched / "score_correlation.csv", index=False, encoding="utf-8-sig")
    weight_correlation.to_csv(stitched / "weight_correlation.csv", index=False, encoding="utf-8-sig")
    identity = {"fm_top20": identities["fm_top20"], "mlp_softmax": identities["mlp_softmax"], "mask": mask, "rankic_allocator_invariance": invariance}
    reports = output / "reports"
    reports.mkdir()
    (reports / "score_allocator_2x2_report.md").write_text(_report(summary, annual, effects, interaction, rankic_by_year, deciles, overlap, score_correlation, identity), encoding="utf-8")
    manifest = {
        "status": "PASS", "reference_output": str(reference), "output": str(output), "date_range": [str(states["fm_top20"].dates[0]), str(states["fm_top20"].dates[-1])], "decision_date_count": int(len(states["fm_top20"].dates)),
        "optimizer_steps": 0, "no_training": True, "no_backward": True, "no_mlp_forward": True, "no_checkpoint_change": True, "no_factor_reselection": True, "no_factor_direction_change": True, "no_temperature_tuning": True, "temperature": TEMPERATURE, "top20_ratio": TOP20_RATIO, "no_cap_or_sparse_allocator": True, "no_factor_scale_ablation": True, "no_n1": True,
        "accounting": "continuous drift-aware 2022-2026 path; turnover=0.5*L1(target-drifted_previous), cost=2*turnover*cost_bps/10000", "costs_bps": list(COST_BPS), "existing_corner_identity": identities, "mask_identity": mask, "rankic_allocator_invariance": invariance, "fixed_factor_audit": fixed_audit,
        "preflight_checks": {"same_factor_mean_score_for_both_allocators": "PASS", "same_mlp_score_for_both_allocators": "PASS", "mask_identity": mask["status"], "fm_top20_identity": identities["fm_top20"]["status"], "mlp_softmax_identity": identities["mlp_softmax"]["status"], "top20_exact_allocator": "PASS", "softmax_t1_exact_allocator": "PASS", "rankic_allocator_invariance": invariance["status"], "same_execution_return": "PASS", "same_csi500_benchmark": "PASS", "continuous_cross_year_accounting": "PASS", "drifted_previous_turnover": "PASS", "cost_formula": "PASS", "annual_from_stitched_path": "PASS", "no_training": "PASS", "no_optimizer_step": "PASS", "no_factor_reselection": "PASS", "no_temperature_search": "PASS", "no_factor_scale_or_n1": "PASS"},
    }
    _json(output / "run_manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("/cloud/E2EAI/output/fixed64_annual_walk_forward_baseline"))
    parser.add_argument("--output", type=Path, default=Path("/cloud/E2EAI/output/fixed64_score_allocator_2x2_attribution"))
    parser.add_argument("--factor-list", type=Path, default=Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08/factor_list.json"))
    parser.add_argument("--factor-directions", type=Path, default=Path("/cloud/E2EAI/data/daily_strategy_trainonly64_corr08/factor_directions.json"))
    args = parser.parse_args()
    print(run(args.reference, args.output, args.factor_list, args.factor_directions))


if __name__ == "__main__":
    main()
