"""Per-image failure taxonomy derived from the canonical IoU matcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from podfa.evaluation.matching import match_detections


@dataclass(frozen=True)
class FailureRecord:
    image_id: str
    failure_type: str
    confidence: float | None
    iou: float
    ground_truth_count: int
    prediction_count: int
    prediction_index: int | None
    ground_truth_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def categorize_failures(
    prediction: dict[str, torch.Tensor],
    target: dict[str, Any],
    confidence_threshold: float,
    iou_threshold: float = 0.5,
    gallery_floor: float = 0.05,
) -> list[FailureRecord]:
    """Assign deterministic nonexclusive records for errors and threshold misses."""
    result = match_detections(prediction, target, confidence_threshold, iou_threshold)
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    ground_truth_count = len(target["boxes"])
    prediction_count = int(((scores >= confidence_threshold) & (labels == 1)).sum().item())
    image_id = str(target.get("source_id", "unknown"))
    matched_ground_truth = {match.ground_truth_index for match in result.matches}
    records: list[FailureRecord] = []

    for prediction_index in result.unmatched_prediction_indices:
        if ground_truth_count:
            row = result.iou_matrix[prediction_index]
            best_iou_tensor, best_gt_tensor = torch.max(row, dim=0)
            best_iou = float(best_iou_tensor.item())
            best_gt = int(best_gt_tensor.item())
        else:
            best_iou, best_gt = 0.0, None
        if best_gt is not None and best_iou >= iou_threshold and best_gt in matched_ground_truth:
            failure_type = "duplicate_detection"
        elif best_iou >= 0.10:
            failure_type = "localization_failure"
        else:
            failure_type = "false_positive"
        records.append(
            FailureRecord(
                image_id=image_id,
                failure_type=failure_type,
                confidence=float(scores[prediction_index].item()),
                iou=best_iou,
                ground_truth_count=ground_truth_count,
                prediction_count=prediction_count,
                prediction_index=prediction_index,
                ground_truth_index=best_gt,
            )
        )

    low_confidence_indices = torch.where(
        (scores >= gallery_floor) & (scores < confidence_threshold) & (labels == 1)
    )[0]
    for prediction_index_tensor in low_confidence_indices:
        prediction_index = int(prediction_index_tensor.item())
        if ground_truth_count == 0:
            continue
        best_iou_tensor, best_gt_tensor = torch.max(result.iou_matrix[prediction_index], dim=0)
        best_iou = float(best_iou_tensor.item())
        if best_iou >= iou_threshold:
            records.append(
                FailureRecord(
                    image_id=image_id,
                    failure_type="low_confidence_correct",
                    confidence=float(scores[prediction_index].item()),
                    iou=best_iou,
                    ground_truth_count=ground_truth_count,
                    prediction_count=prediction_count,
                    prediction_index=prediction_index,
                    ground_truth_index=int(best_gt_tensor.item()),
                )
            )

    for ground_truth_index in result.unmatched_ground_truth_indices:
        records.append(
            FailureRecord(
                image_id=image_id,
                failure_type="false_negative",
                confidence=None,
                iou=0.0,
                ground_truth_count=ground_truth_count,
                prediction_count=prediction_count,
                prediction_index=None,
                ground_truth_index=ground_truth_index,
            )
        )
    return records
