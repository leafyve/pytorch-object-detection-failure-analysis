"""Detector factory with explicit transfer-learning metadata."""

from __future__ import annotations

from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

SUPPORTED_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_fpn"
PRETRAINED_WEIGHTS_NAME = "COCO_V1"


class ModelFactoryError(ValueError):
    """Raised for unsupported or unsafe detector factory inputs."""


def build_model(
    architecture: str,
    num_classes: int,
    pretrained: bool,
    min_size: int,
    max_size: int,
) -> GeneralizedRCNN:
    """Build Faster R-CNN and replace its COCO classifier for person detection."""
    if architecture != SUPPORTED_ARCHITECTURE:
        raise ModelFactoryError(f"unsupported architecture: {architecture}")
    if num_classes != 2:
        raise ModelFactoryError("num_classes must be 2 for background and person")
    if min_size <= 0 or max_size < min_size:
        raise ModelFactoryError("model sizes must satisfy 0 < min_size <= max_size")
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=3 if pretrained else None,
        min_size=min_size,
        max_size=max_size,
    )
    predictor = model.roi_heads.box_predictor
    if not hasattr(predictor, "cls_score"):
        raise ModelFactoryError("unexpected Torchvision predictor without cls_score")
    in_features = predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.architecture_name = architecture
    model.pretrained_weights_name = PRETRAINED_WEIGHTS_NAME if pretrained else "none"
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the number of parameters updated by the optimizer."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
