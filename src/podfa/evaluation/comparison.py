"""Validation-only experiment comparison and model selection."""

from __future__ import annotations

from typing import Any

METRICS = (
    ("map_50_95", "mAP@0.50:0.95"),
    ("map_50", "mAP@0.50"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("true_positives", "True positives"),
    ("false_positives", "False positives"),
    ("false_negatives", "False negatives"),
)


def compare_experiments(
    baseline: dict[str, Any], improved: dict[str, Any]
) -> dict[str, Any]:
    """Compare the same validation split and select by the documented tie-breaks."""
    for key, _ in METRICS:
        if key not in baseline or key not in improved:
            raise ValueError(f"both experiment metrics must include {key}")
    baseline_key = (baseline["map_50_95"], baseline["map_50"], baseline["recall"])
    improved_key = (improved["map_50_95"], improved["map_50"], improved["recall"])
    preferred = "improved" if improved_key > baseline_key else "baseline"
    return {
        "selection_split": "validation",
        "selection_rule": "mAP@0.50:0.95, then mAP@0.50, then recall",
        "preferred_experiment": preferred,
        "metrics": {
            key: {
                "label": label,
                "baseline": baseline[key],
                "improved": improved[key],
                "delta": round(float(improved[key]) - float(baseline[key]), 10),
            }
            for key, label in METRICS
        },
    }


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render a compact recruiter-readable comparison with honest negative deltas."""
    rows = [
        "# Experiment comparison",
        "",
        "Both experiments use the same seed-42 train/validation split. The preferred model was "
        "selected using validation data only; the held-out test split was not consulted.",
        "",
        "| Metric | Baseline | Improved | Delta |",
        "|---|---:|---:|---:|",
    ]
    count_keys = {"true_positives", "false_positives", "false_negatives"}
    for key, values in comparison["metrics"].items():
        if key in count_keys:
            baseline = f"{int(values['baseline'])}"
            improved = f"{int(values['improved'])}"
            delta = f"{int(values['delta']):+d}"
        else:
            baseline = f"{values['baseline']:.4f}"
            improved = f"{values['improved']:.4f}"
            delta = f"{values['delta']:+.4f}"
        rows.append(f"| {values['label']} | {baseline} | {improved} | {delta} |")
    rows.extend(
        [
            "",
            f"**Preferred experiment:** `{comparison['preferred_experiment']}`.",
            "",
            f"Selection rule: {comparison['selection_rule']}.",
            "",
        ]
    )
    return "\n".join(rows)
