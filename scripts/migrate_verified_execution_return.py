"""Copy the previously verified execution-return array before legacy-panel deletion."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Existing verified source array. This migration utility has no legacy default.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/cloud/E2EAI/data/execution_returns/execution_1d_return_verified.npy"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/daily_strategy_diagnostics/execution_return_migration_verification.json"),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify an already copied destination without replacing it.",
    )
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.destination.exists() and not args.verify_existing:
        raise FileExistsError(f"Refusing to overwrite destination: {args.destination}")
    copied = False
    if not args.destination.exists():
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source, args.destination)
        copied = True
    source = np.load(args.source, mmap_mode="r", allow_pickle=False)
    destination = np.load(args.destination, mmap_mode="r", allow_pickle=False)
    if source.shape != destination.shape or source.dtype != destination.dtype:
        raise RuntimeError("Copied execution return has incompatible shape or dtype")
    generator = np.random.default_rng(42)
    rows = generator.integers(0, source.shape[0], size=200)
    columns = generator.integers(0, source.shape[1], size=200)
    sampled_equal = bool(
        np.array_equal(source[rows, columns], destination[rows, columns], equal_nan=True)
    )
    source_hash = _sha256(args.source)
    destination_hash = _sha256(args.destination)
    report = {
        "source": str(args.source),
        "destination": str(args.destination),
        "copied_this_invocation": copied,
        "source_sha256": source_hash,
        "destination_sha256": destination_hash,
        "sha256_equal": source_hash == destination_hash,
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "random_sample_count": 200,
        "random_samples_equal": sampled_equal,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["sha256_equal"] or not sampled_equal:
        raise RuntimeError("Execution-return migration verification failed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
