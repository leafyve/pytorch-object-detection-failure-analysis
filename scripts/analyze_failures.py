"""Rebuild structured failure evidence from a saved prediction bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podfa.evaluation.evaluator import load_prediction_bundle, write_metrics
from podfa.evaluation.visualize import analyze_prediction_bundle, write_failure_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions, targets = load_prediction_bundle(args.predictions)
    failures, counts = analyze_prediction_bundle(predictions, targets, args.threshold)
    write_failure_csv(failures, args.output_csv)
    write_metrics(
        {"confidence_threshold": args.threshold, "counts": counts}, args.summary_json
    )
    print(json.dumps({"failure_records": len(failures), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
