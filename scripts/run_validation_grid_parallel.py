"""Run the eight unique one-factor-at-a-time validation experiments in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run_id(row: dict) -> str:
    if len(row.get("families", [])) > 1:
        return "reference_theta10_gamma1N_lambda010"
    family = row["family"]
    value = row["value"]
    token = str(value).replace(".", "p")
    return f"{family}_{token}"


def _run_one(project: Path, output_root: Path, row: dict, epochs: int, patience: int) -> dict:
    run_id = _run_id(row)
    destination = output_root / run_id
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "validation_metrics.json"
    if metrics_path.exists():
        return {"run_id": run_id, "status": "skipped_existing"}
    command = [
        sys.executable, "-m", "scripts.validation_only_train",
        "--run-id", run_id,
        "--output-root", str(output_root),
        "--theta", str(row["theta"]),
        "--gamma-p", str(row["gamma_p"]),
        "--gamma-p-mode", row["gamma_p_mode"],
        "--lambda-s", str(row["lambda_s"]),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--device", "cuda",
    ]
    with (destination / "process.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT)
    return {"run_id": run_id, "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode}


def _aggregate(output_root: Path, experiments: list[dict]) -> None:
    summary_rows: list[dict] = []
    horizon_rows: list[dict] = []
    for row in experiments:
        run_id = _run_id(row)
        path = output_root / run_id / "validation_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        experiment = metrics["experiment"]
        summary_rows.append({
            **experiment,
            "families": ",".join(row.get("families", [row["family"]])),
            **metrics["losses"],
        })
        for horizon, values in metrics["by_horizon"].items():
            horizon_rows.append({
                "run_id": run_id,
                "families": ",".join(row.get("families", [row["family"]])),
                "theta": experiment["theta"],
                "gamma_p": experiment["gamma_p"],
                "gamma_p_mode": experiment["gamma_p_mode"],
                "lambda_s": experiment["lambda_s"],
                "horizon": int(horizon),
                **values,
            })
    summary = pd.DataFrame(summary_rows)
    horizons = pd.DataFrame(horizon_rows)
    summary.to_csv(output_root / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_parquet(output_root / "experiment_summary.parquet", index=False)
    horizons.to_csv(output_root / "validation_summary_by_horizon.csv", index=False, encoding="utf-8-sig")
    horizons.to_parquet(output_root / "validation_summary_by_horizon.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/validation_sensitivity/csi500/fold_0"))
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_experiments = manifest["experiments"]
    experiments = all_experiments[args.start_index:args.end_index]
    project = Path(__file__).resolve().parents[1]
    args.output_root.mkdir(parents=True, exist_ok=True)
    statuses: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = [
            pool.submit(_run_one, project, args.output_root, row, args.epochs, args.patience)
            for row in experiments
        ]
        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            statuses.append(status)
            (args.output_root / "parallel_status.json").write_text(
                json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(status, ensure_ascii=False), flush=True)
    failed = [status for status in statuses if status.get("status") == "failed"]
    if failed:
        raise SystemExit(f"Experiments failed: {failed}")
    _aggregate(args.output_root, all_experiments)


if __name__ == "__main__":
    main()
