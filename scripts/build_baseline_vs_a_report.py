"""Write a validation-only comparison of Factor Mean and Experiment A.

This intentionally consumes only artifacts from the train-only 64-factor panel.
It neither reads the OOS split nor starts another experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: object, *, digits: int = 6) -> str:
    if value is None:
        return "—"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def _metrics_row(values: dict[str, Any], *, a: bool) -> list[tuple[str, str]]:
    result = [
        ("Validation execution RankIC", _number(values.get("validation_execution_rankic"))),
        ("Validation execution ICIR", _number(values.get("validation_execution_icir"))),
        (
            "Positive execution RankIC ratio",
            _number(values.get("validation_execution_positive_ic_ratio")),
        ),
        ("0 bps daily Sharpe", _number(values.get("validation_daily_sharpe_cost_0p0bps"))),
        ("5 bps daily Sharpe", _number(values.get("validation_daily_sharpe_cost_5p0bps"))),
        ("10 bps daily Sharpe", _number(values.get("validation_daily_sharpe_cost_10p0bps"))),
        (
            "5 bps annualized daily return",
            _number(values.get("validation_daily_return_cost_5p0bps")),
        ),
        ("5 bps daily maximum drawdown", _number(values.get("validation_daily_mdd_cost_5p0bps"))),
        ("Daily turnover", _number(values.get("validation_daily_turnover"))),
        ("Effective holdings", _number(values.get("validation_effective_n"), digits=2)),
        ("Maximum single-stock weight", _number(values.get("validation_max_weight"))),
        ("Selected stock count", _number(values.get("validation_selected_stock_count"), digits=2)),
    ]
    if a:
        result.extend(
            [
                ("Best epoch (zero based)", _number(values.get("best_epoch"), digits=0)),
                ("Completed epochs", _number(values.get("epochs_completed"), digits=0)),
                ("Early stopped", str(values.get("stopped_early", False))),
                ("Best validation target", _number(values.get("best_metric"))),
            ]
        )
    return result


def _best_training_row(experiment_dir: Path, best_epoch: object) -> dict[str, Any] | None:
    history = experiment_dir / "training.jsonl"
    if not history.is_file() or best_epoch is None:
        return None
    expected = int(best_epoch)
    for line in history.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row.get("epoch", -1)) == expected:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment-a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = _read_json(args.baseline / "validation_metrics.json")
    experiment_a = _read_json(args.experiment_a / "validation_metrics.json")
    if baseline.get("oos_evaluated") is not False or experiment_a.get("oos_evaluated") is not False:
        raise ValueError("Comparison accepts validation-only artifacts only")

    a_train = _best_training_row(args.experiment_a, experiment_a.get("best_epoch"))
    train_checkpoint_path = args.experiment_a / "train_best_checkpoint_metrics.json"
    train_checkpoint = _read_json(train_checkpoint_path) if train_checkpoint_path.is_file() else {}
    lines = [
        "# Factor Mean baseline vs Experiment A",
        "",
        "**Scope.** Validation only: 2021-01-04 to 2021-12-03 after a 20-trading-day label-exit purge. "
        "2022 test split not evaluated; no 2022 OOS result is read or reported.",
        "",
        "| Metric | Factor Mean Top20 | Experiment A |",
        "|---|---:|---:|",
    ]
    baseline_map = dict(_metrics_row(baseline, a=False))
    experiment_map = dict(_metrics_row(experiment_a, a=True))
    for name in list(baseline_map) + [key for key in experiment_map if key not in baseline_map]:
        lines.append(f"| {name} | {baseline_map.get(name, '—')} | {experiment_map.get(name, '—')} |")

    lines.extend(["", "## Experiment A best-checkpoint training record", ""])
    if a_train is None:
        lines.append("Best-epoch training record was unavailable.")
    else:
        train_rankic = train_checkpoint.get("rank_ic_deep_factor_2_execution_1d")
        lines.extend(
            [
                f"- Training mean execution RankIC: {_number(train_rankic)}",
                f"- Training optimization mean IC (all supervised horizons): {_number(a_train.get('train_mean_ic'))}",
                f"- Validation execution RankIC: {_number(experiment_a.get('validation_execution_rankic'))}",
                f"- Validation execution ICIR: {_number(experiment_a.get('validation_execution_icir'))}",
                f"- Validation positive RankIC ratio: {_number(experiment_a.get('validation_execution_positive_ic_ratio'))}",
            ]
        )
    lines.extend(
        [
            "",
            "## Audit notes",
            "",
            "- Both outputs were produced from the train-only 64-factor panel and verified execution return series.",
            "- This file is descriptive only; it does not select a deployment model or claim 2022 performance.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
