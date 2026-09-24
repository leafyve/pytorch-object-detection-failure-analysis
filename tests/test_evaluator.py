from __future__ import annotations

import json
from pathlib import Path

import torch

from podfa.evaluation.evaluator import (
    evaluate_predictions,
    select_confidence_threshold,
    write_metrics,
)
from podfa.evaluation.visualize import render_detection


def _prediction(boxes: list[list[float]], scores: list[float]) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores),
        "labels": torch.ones(len(boxes), dtype=torch.int64),
    }


def _target(boxes: list[list[float]]) -> dict[str, object]:
    tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return {
        "boxes": tensor,
        "labels": torch.ones(len(boxes), dtype=torch.int64),
        "source_id": "image-1",
    }


def test_threshold_selection_maximizes_f1_using_validation_predictions() -> None:
    predictions = [_prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [0.4, 0.3])]
    targets = [_target([[0, 0, 10, 10]])]

    selection = select_confidence_threshold(predictions, targets, [0.25, 0.35, 0.45])

    assert selection.threshold == 0.35
    assert selection.metrics["f1"] == 1.0
    assert len(selection.curve) == 3


def test_evaluate_predictions_and_write_metrics_include_required_keys(tmp_path: Path) -> None:
    predictions = [_prediction([[0, 0, 10, 10]], [0.9])]
    targets = [_target([[0, 0, 10, 10]])]
    metrics = evaluate_predictions(predictions, targets, confidence_threshold=0.5)
    output = tmp_path / "metrics.json"
    write_metrics(metrics, output)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert {
        "map_50",
        "map_50_95",
        "precision",
        "recall",
        "true_positives",
        "false_positives",
        "false_negatives",
        "confidence_threshold",
    } <= saved.keys()


def test_render_detection_does_not_mutate_inputs() -> None:
    image = torch.zeros(3, 32, 32)
    prediction = _prediction([[2, 2, 20, 20]], [0.9])
    target = _target([[1, 1, 21, 21]])
    before = prediction["boxes"].clone()

    rendered = render_detection(image, target, prediction, confidence_threshold=0.5)

    assert rendered.size == (32, 32)
    assert torch.equal(prediction["boxes"], before)
    assert rendered.getpixel((1, 1)) != (0, 0, 0)
