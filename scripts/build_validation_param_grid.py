"""Expand the documented validation-only sensitivity grid into a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_grid(spec: dict) -> list[dict]:
    reference = dict(spec["reference"])
    rows: list[dict] = []
    for family, settings in spec["families"].items():
        for value in settings["values"]:
            row = {
                **reference,
                "family": family,
                "value": float(value),
                "split": "purged_validation_only",
            }
            if family == "theta":
                row["theta"] = float(value)
            elif family == "gamma_p":
                row["gamma_p_mode"] = settings.get("mode", "inverse_n")
                row["gamma_p"] = float(value)
            elif family == "lambda_s":
                row["lambda_s"] = float(value)
            rows.append(row)
    unique: dict[tuple, dict] = {}
    for row in rows:
        key = (row["theta"], row["gamma_p_mode"], row["gamma_p"], row["lambda_s"])
        if key not in unique:
            unique[key] = row
            unique[key]["families"] = []
        unique[key]["families"].append(row["family"])
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path(__file__).parents[1] / "configs/validation_param_grid.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    manifest = {
        "policy": spec["policy"],
        "grid_size": len(build_grid(spec)),
        "experiments": build_grid(spec),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest['grid_size']} validation-only experiments to {args.output}")


if __name__ == "__main__":
    main()
