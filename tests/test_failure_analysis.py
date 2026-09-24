from __future__ import annotations

import torch

from podfa.evaluation.failure_analysis import categorize_failures


def _prediction(boxes: list[list[float]], scores: list[float]) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores),
        "labels": torch.ones(len(boxes), dtype=torch.int64),
    }


def _target(boxes: list[list[float]]) -> dict[str, object]:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.ones(len(boxes), dtype=torch.int64),
        "source_id": "fixture-1",
    }


def test_duplicate_has_precedence_over_generic_false_positive() -> None:
    failures = categorize_failures(
        _prediction([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8]),
        _target([[0, 0, 10, 10]]),
        confidence_threshold=0.5,
    )

    assert [failure.failure_type for failure in failures] == ["duplicate_detection"]
    assert failures[0].iou == 1.0


def test_localization_failure_and_missed_ground_truth_are_both_recorded() -> None:
    failures = categorize_failures(
        _prediction([[5, 0, 15, 10]], [0.9]),
        _target([[0, 0, 10, 10]]),
        confidence_threshold=0.5,
    )

    assert {failure.failure_type for failure in failures} == {
        "localization_failure",
        "false_negative",
    }


def test_low_confidence_geometric_match_is_separate_from_thresholded_miss() -> None:
    failures = categorize_failures(
        _prediction([[0, 0, 10, 10]], [0.4]),
        _target([[0, 0, 10, 10]]),
        confidence_threshold=0.5,
        gallery_floor=0.2,
    )

    assert {failure.failure_type for failure in failures} == {
        "low_confidence_correct",
        "false_negative",
    }


def test_unrelated_prediction_is_false_positive() -> None:
    failures = categorize_failures(
        _prediction([[20, 20, 30, 30]], [0.9]),
        _target([[0, 0, 10, 10]]),
        confidence_threshold=0.5,
    )

    assert {failure.failure_type for failure in failures} == {"false_positive", "false_negative"}
