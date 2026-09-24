"""Run A/B/C1/C2/D in dependency order and aggregate validation summaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["experiment_a", "experiment_b", "experiment_c1", "experiment_c2", "experiment_d"],
    )
    parser.add_argument("--fold-config", default="configs/rolling_fold0_2018_2022.yaml")
    parser.add_argument("--output-root", type=Path, default=Path("output/daily_strategy"))
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    summaries: list[pd.DataFrame] = []
    for value in args.experiments:
        experiment = value.lower()
        if not experiment.startswith("experiment_"):
            experiment = f"experiment_{experiment}"
        config = Path("configs/experiments") / f"{experiment}.yaml"
        output = args.output_root / experiment
        command = [
            sys.executable,
            "-m",
            "scripts.run_daily_experiment",
            "--config",
            str(config),
            "--fold-config",
            args.fold_config,
            "--output",
            str(output),
        ]
        if args.synthetic_smoke:
            command.append("--synthetic-smoke")
        subprocess.run(command, check=True)
        summaries.append(pd.read_csv(output / "summary.csv"))
    if summaries:
        args.output_root.mkdir(parents=True, exist_ok=True)
        pd.concat(summaries, ignore_index=True).to_csv(
            args.output_root / "summary.csv", index=False, encoding="utf-8-sig"
        )


if __name__ == "__main__":
    main()
