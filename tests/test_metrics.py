from __future__ import annotations

import pytest
import torch

from podfa.evaluation.metrics import compute_coco_metrics, compute_detection_metrics


def _target(boxes: list[list[float]]) -> dict[str, torch.Tensor]:
    tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return {
        "boxes": tensor,
        "labels": torch.ones((len(boxes),), dtype=torch.int64),
        "image_id": torch.tensor(0),
        "area": (tensor[:, 2] - tensor[:, 0]) * (tensor[:, 3] - tensor[:, 1]),
        "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
    }


def _prediction(
    boxes: list[list[float]], scores: list[float]
) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.ones((len(boxes),), dtype=torch.int64),
    }


@pytest.mark.parametrize(
    ("prediction", "target", "expected"),
    [
        (_prediction([[0, 0, 10, 10]], [0.9]), _target([[0, 0, 10, 10]]), (1, 0, 0)),
        (_prediction([], []), _target([[0, 0, 10, 10]]), (0, 0, 1)),
        (_prediction([[20, 20, 30, 30]], [0.9]), _target([[0, 0, 10, 10]]), (0, 1, 1)),
        (_prediction([], []), _target([]), (0, 0, 0)),
    ],
)
def test_detection_counts_match_hand_calculation(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    expected: tuple[int, int, int],
) -> None:
    metrics = compute_detection_metrics([prediction], [target], confidence_threshold=0.5)

    observed = (
        metrics["true_positives"],
        metrics["false_positives"],
        metrics["false_negatives"],
    )
    assert observed == expected


def test_precision_recall_and_f1_are_derived_from_counts() -> None:
    predictions = [
        _prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8]),
        _prediction([], []),
    ]
    targets = [_target([[0, 0, 10, 10]]), _target([[5, 5, 15, 15]])]

    metrics = compute_detection_metrics(predictions, targets, confidence_threshold=0.5)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_perfect_prediction_has_near_one_coco_map() -> None:
    metrics = compute_coco_metrics(
        [_prediction([[0, 0, 10, 10]], [0.99])], [_target([[0, 0, 10, 10]])]
    )

    assert metrics["map_50"] > 0.99
    assert metrics["map_50_95"] > 0.99
