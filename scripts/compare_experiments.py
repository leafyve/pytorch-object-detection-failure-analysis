"""Compare validation runs and freeze the preferred checkpoint before test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from podfa.evaluation.comparison import compare_experiments, render_comparison_markdown


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--improved-metrics", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--improved-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--improved-checkpoint", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    improved = json.loads(args.improved_metrics.read_text(encoding="utf-8"))
    comparison = compare_experiments(baseline, improved)
    args.comparison_report.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_report.write_text(
        render_comparison_markdown(comparison), encoding="utf-8"
    )
    preferred = comparison["preferred_experiment"]
    metrics = improved if preferred == "improved" else baseline
    config = args.improved_config if preferred == "improved" else args.baseline_config
    checkpoint = (
        args.improved_checkpoint if preferred == "improved" else args.baseline_checkpoint
    )
    selection = {
        **comparison,
        "preferred_config": str(config),
        "preferred_checkpoint": str(checkpoint),
        "preferred_checkpoint_sha256": _sha256(checkpoint),
        "selected_confidence_threshold": metrics["confidence_threshold"],
        "validation_metrics": metrics,
        "test_split_used_for_selection": False,
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
