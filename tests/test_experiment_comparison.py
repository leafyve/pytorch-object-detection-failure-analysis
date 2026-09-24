from __future__ import annotations

from podfa.evaluation.comparison import compare_experiments, render_comparison_markdown


def test_compare_experiments_selects_validation_map_then_documents_deltas() -> None:
    baseline = {
        "map_50_95": 0.7,
        "map_50": 0.9,
        "precision": 0.8,
        "recall": 0.85,
        "f1": 0.824,
        "true_positives": 17,
        "false_positives": 4,
        "false_negatives": 3,
    }
    improved = {
        "map_50_95": 0.75,
        "map_50": 0.88,
        "precision": 0.82,
        "recall": 0.86,
        "f1": 0.84,
        "true_positives": 18,
        "false_positives": 4,
        "false_negatives": 2,
    }

    comparison = compare_experiments(baseline, improved)
    markdown = render_comparison_markdown(comparison)

    assert comparison["preferred_experiment"] == "improved"
    assert comparison["metrics"]["map_50_95"]["delta"] == 0.05
    assert "| mAP@0.50:0.95 | 0.7000 | 0.7500 | +0.0500 |" in markdown
    assert "selected using validation data only" in markdown
