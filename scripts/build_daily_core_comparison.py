"""Create the formal corr08 Factor Mean versus Experiment A comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads((path / "validation_metrics.json").read_text(encoding="utf-8"))


def _value(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build(root: Path) -> Path:
    baseline = _load(root / "factor_mean_baseline")
    experiment = _load(root / "experiment_a")
    rows = [
        ("train mean RankIC (A only)", "train_mean_execution_rankic"),
        ("IC gap train - validation (A only)", "ic_gap_train_minus_validation"),
        ("validation mean RankIC", "validation_execution_rankic"),
        ("validation ICIR", "validation_execution_icir"),
        ("positive IC ratio", "validation_execution_positive_ic_ratio"),
        ("0bps Sharpe", "validation_daily_sharpe_cost_0p0bps"),
        ("5bps Sharpe", "validation_daily_sharpe_cost_5p0bps"),
        ("7.5bps Sharpe", "validation_daily_sharpe_cost_7p5bps"),
        ("10bps Sharpe", "validation_daily_sharpe_cost_10p0bps"),
        ("7.5bps annual return", "validation_daily_return_cost_7p5bps"),
        ("7.5bps MDD", "validation_daily_mdd_cost_7p5bps"),
        ("turnover", "validation_daily_turnover"),
        ("effective N", "validation_effective_n"),
        ("7.5bps annual excess return", "validation_active_annual_excess_return_cost_7p5bps"),
        ("7.5bps tracking error", "validation_tracking_error_cost_7p5bps"),
        ("7.5bps information ratio", "validation_information_ratio_cost_7p5bps"),
        ("7.5bps active max drawdown", "validation_active_max_drawdown_cost_7p5bps"),
        ("7.5bps active win rate", "validation_active_win_rate_cost_7p5bps"),
        ("daily score rank correlation", "validation_daily_score_rank_correlation"),
        ("Top20 next-day retention", "validation_top20_next_day_retention"),
    ]
    lines = [
        "# corr08 pure daily_core: Factor Mean vs Experiment A",
        "",
        "- Validation only: 2021-01-04 to 2021-12-29 after h=2 purge.",
        "- Signal timing: decision t, VWAP entry t+1, VWAP exit/valuation t+2.",
        "- Factor selection remains the frozen corr08 train-only manifest (2018-01-02 to 2020-12-03).",
        "- Experiment A is a signal sanity check only; no FactorSelector/GAT/portfolio E2E loss is active.",
        "- 2022 test split was not evaluated.",
        "",
        "| Metric | Factor Mean | Experiment A |",
        "|---|---:|---:|",
    ]
    for label, key in rows:
        lines.append(f"| {label} | {_value(baseline, key)} | {_value(experiment, key)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The table is descriptive and does not trigger any subsequent B/C/D experiment.",
            "Compare gross and 7.5bp net Sharpe together with turnover and score stability before deciding whether to enter the differentiable portfolio stage.",
        ]
    )
    target = root / "baseline_vs_a.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.root))


if __name__ == "__main__":
    main()
