"""Paired image/target transforms that preserve box semantics."""

from __future__ import annotations

from typing import Any

import torch


class DetectionTransform:
    """Apply deterministic-safe tensor augmentations to image/target pairs."""

    def __init__(self, *, training: bool, horizontal_flip_probability: float = 0.0) -> None:
        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between 0 and 1")
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability

    def __call__(
        self, image: torch.Tensor, target: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        transformed = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        if self.training and torch.rand(()) < self.horizontal_flip_probability:
            image = torch.flip(image, dims=(-1,))
            boxes = transformed["boxes"]
            width = image.shape[-1]
            x_min = width - boxes[:, 2]
            x_max = width - boxes[:, 0]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
            if "masks" in transformed:
                transformed["masks"] = torch.flip(transformed["masks"], dims=(-1,))
        return image, transformed


def build_transforms(training: bool, experiment: str) -> DetectionTransform:
    """Build the named experiment transform policy."""
    if experiment not in {"baseline", "smoke", "improved"}:
        raise ValueError(f"unknown transform experiment: {experiment}")
    flip_probability = 0.5 if training and experiment != "smoke" else 0.0
    return DetectionTransform(
        training=training,
        horizontal_flip_probability=flip_probability,
    )
