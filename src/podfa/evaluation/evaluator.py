"""Prediction collection, validation threshold selection, and serialization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from podfa.evaluation.metrics import compute_coco_metrics, compute_detection_metrics


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metrics: dict[str, int | float]
    curve: tuple[dict[str, int | float], ...]


def _cpu_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def collect_predictions(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Run inference and detach all outputs for deterministic CPU evaluation."""
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            predictions.extend(
                [
                    {key: value.detach().cpu() for key, value in output.items()}
                    for output in outputs
                ]
            )
            targets.extend([_cpu_target(target) for target in batch_targets])
    if not targets:
        raise ValueError("evaluation loader yielded no samples")
    return predictions, targets


def select_confidence_threshold(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
    candidates: list[float],
    iou_threshold: float = 0.5,
) -> ThresholdSelection:
    """Select validation threshold by F1, then recall, precision, and threshold."""
    if not candidates:
        raise ValueError("at least one threshold candidate is required")
    if any(not 0 <= candidate <= 1 for candidate in candidates):
        raise ValueError("threshold candidates must be between 0 and 1")
    curve = tuple(
        compute_detection_metrics(predictions, targets, candidate, iou_threshold)
        for candidate in sorted(set(candidates))
    )
    selected = max(
        curve,
        key=lambda metrics: (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["confidence_threshold"],
        ),
    )
    return ThresholdSelection(
        threshold=float(selected["confidence_threshold"]),
        metrics=dict(selected),
        curve=curve,
    )


def evaluate_predictions(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
    confidence_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, int | float]:
    """Combine confidence-independent COCO AP and thresholded operating metrics."""
    metrics = compute_detection_metrics(
        predictions, targets, confidence_threshold, iou_threshold
    )
    metrics.update(compute_coco_metrics(predictions, targets))
    metrics["evaluated_images"] = len(targets)
    return metrics


def write_metrics(metrics: dict[str, Any], path: Path | str) -> None:
    """Atomically write machine-readable metric evidence."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def save_prediction_bundle(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
    path: Path | str,
) -> None:
    """Persist reusable validation/test predictions outside normal Git history."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({"predictions": predictions, "targets": targets}, temporary)
    os.replace(temporary, output)


def load_prediction_bundle(
    path: Path | str,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Load prediction evidence produced locally by this pipeline."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {"predictions", "targets"}:
        raise ValueError(f"malformed prediction bundle: {path}")
    return payload["predictions"], payload["targets"]
