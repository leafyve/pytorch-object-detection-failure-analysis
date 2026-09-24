"""Thresholded counts and COCO-style mean average precision."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from podfa.evaluation.matching import match_detections


def compute_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    confidence_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, int | float]:
    """Aggregate TP/FP/FN, precision, recall, and F1 across images."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    true_positives = false_positives = false_negatives = 0
    for prediction, target in zip(predictions, targets, strict=True):
        result = match_detections(prediction, target, confidence_threshold, iou_threshold)
        true_positives += result.true_positives
        false_positives += result.false_positives
        false_negatives += result.false_negatives
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _xyxy_to_xywh(box: torch.Tensor) -> list[float]:
    x_min, y_min, x_max, y_max = (float(value) for value in box.tolist())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def compute_coco_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compute COCO AP for the person class at IoU 0.50 and 0.50:0.95."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, (prediction, target) in enumerate(zip(predictions, targets, strict=True), 1):
        target_boxes = target["boxes"].detach().cpu()
        max_coordinate = float(target_boxes.max().item()) if target_boxes.numel() else 1.0
        images.append(
            {
                "id": image_id,
                "width": int(np.ceil(max_coordinate + 1)),
                "height": int(np.ceil(max_coordinate + 1)),
            }
        )
        for box in target_boxes:
            xywh = _xyxy_to_xywh(box)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        for box, score, label in zip(
            prediction["boxes"].detach().cpu(),
            prediction["scores"].detach().cpu(),
            prediction["labels"].detach().cpu(),
            strict=True,
        ):
            if int(label.item()) == 1:
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": _xyxy_to_xywh(box),
                        "score": float(score.item()),
                    }
                )
    if not annotations or not detections:
        return {"map_50_95": 0.0, "map_50": 0.0}
    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
        results = ground_truth.loadRes(detections)
        evaluator = COCOeval(ground_truth, results, "bbox")
        evaluator.params.imgIds = [image["id"] for image in images]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"map_50_95": float(evaluator.stats[0]), "map_50": float(evaluator.stats[1])}
