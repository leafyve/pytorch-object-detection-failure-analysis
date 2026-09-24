"""Evaluate a trained checkpoint on validation or one held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podfa.evaluation.run import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--failure-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        confidence_threshold=args.threshold,
        select_threshold=args.select_threshold,
        failure_csv_path=args.failure_csv,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
