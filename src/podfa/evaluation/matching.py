"""Deterministic score-ordered IoU matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import box_iou


@dataclass(frozen=True)
class DetectionMatch:
    prediction_index: int
    ground_truth_index: int
    iou: float
    score: float


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[DetectionMatch, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_ground_truth_indices: tuple[int, ...]
    iou_matrix: torch.Tensor

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_prediction_indices)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_ground_truth_indices)


def box_iou_matrix(prediction_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU while supporting empty box collections."""
    if prediction_boxes.ndim != 2 or prediction_boxes.shape[-1] != 4:
        raise ValueError("prediction boxes must have shape [N, 4]")
    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise ValueError("target boxes must have shape [N, 4]")
    return box_iou(prediction_boxes.float(), target_boxes.float())


def match_detections(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    confidence_threshold: float,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedily match score-sorted person predictions to unique ground truths."""
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    boxes = prediction["boxes"].detach().cpu()
    target_boxes = target["boxes"].detach().cpu()
    eligible = torch.where((scores >= confidence_threshold) & (labels == 1))[0]
    eligible = eligible[torch.argsort(scores[eligible], descending=True, stable=True)]
    ious = box_iou_matrix(boxes, target_boxes)
    unmatched_ground_truth = set(range(len(target_boxes)))
    matches: list[DetectionMatch] = []
    unmatched_predictions: list[int] = []

    for prediction_index_tensor in eligible:
        prediction_index = int(prediction_index_tensor.item())
        if not unmatched_ground_truth:
            unmatched_predictions.append(prediction_index)
            continue
        available = sorted(unmatched_ground_truth)
        available_ious = ious[prediction_index, available]
        best_position = int(torch.argmax(available_ious).item())
        best_ground_truth = available[best_position]
        best_iou = float(available_ious[best_position].item())
        if best_iou >= iou_threshold:
            matches.append(
                DetectionMatch(
                    prediction_index=prediction_index,
                    ground_truth_index=best_ground_truth,
                    iou=best_iou,
                    score=float(scores[prediction_index].item()),
                )
            )
            unmatched_ground_truth.remove(best_ground_truth)
        else:
            unmatched_predictions.append(prediction_index)

    return MatchResult(
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(unmatched_predictions),
        unmatched_ground_truth_indices=tuple(sorted(unmatched_ground_truth)),
        iou_matrix=ious,
    )
