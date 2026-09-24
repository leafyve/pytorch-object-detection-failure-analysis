"""Torchvision-compatible Penn-Fudan detection dataset."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from podfa.data.splits import SPLIT_NAMES, SplitManifest, load_split_manifest

Target = dict[str, Any]
DetectionTransforms = Callable[[torch.Tensor, Target], tuple[torch.Tensor, Target]]


class BoundingBoxError(ValueError):
    """Raised when target boxes cannot represent valid image regions."""


class DatasetItemError(ValueError):
    """Raised when an image/mask pair cannot produce a valid target."""


def validate_boxes(boxes: torch.Tensor, width: int, height: int) -> None:
    """Validate finite XYXY boxes with positive area inside the image."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise BoundingBoxError(f"boxes must have shape [N, 4], received {tuple(boxes.shape)}")
    if not torch.isfinite(boxes).all():
        raise BoundingBoxError("boxes must contain only finite coordinates")
    if boxes.numel() == 0:
        raise BoundingBoxError("target must contain at least one bounding box")
    if ((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).any():
        raise BoundingBoxError("boxes must have positive area")
    if (
        (boxes[:, 0] < 0)
        | (boxes[:, 1] < 0)
        | (boxes[:, 2] > width)
        | (boxes[:, 3] > height)
    ).any():
        raise BoundingBoxError(
            f"boxes must stay within image bounds width={width}, height={height}"
        )


class PennFudanDataset(Dataset[tuple[torch.Tensor, Target]]):
    """Load Penn-Fudan records from one deterministic manifest partition."""

    def __init__(
        self,
        root: Path | str,
        manifest: SplitManifest | Path | str,
        split: str,
        transforms: DetectionTransforms | None = None,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, received {split!r}")
        self.root = Path(root)
        self.manifest = (
            load_split_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        )
        self.split = split
        self.entries = self.manifest.partitions[split]
        self.transforms = transforms
        if not self.entries:
            raise DatasetItemError(f"split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.entries)

    def get_height_and_width(self, index: int) -> tuple[int, int]:
        entry = self.entries[index]
        with Image.open(self.root / "PNGImages" / entry.image_file) as image:
            width, height = image.size
        return height, width

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Target]:
        entry = self.entries[index]
        image_path = self.root / "PNGImages" / entry.image_file
        mask_path = self.root / "PedMasks" / entry.mask_file
        try:
            with Image.open(image_path) as source_image:
                image = pil_to_tensor(source_image.convert("RGB")).float().div(255.0)
            with Image.open(mask_path) as source_mask:
                mask_array = np.array(source_mask, dtype=np.int64, copy=True)
        except (OSError, UnidentifiedImageError) as exc:
            raise DatasetItemError(f"failed to read pair for {entry.image_id}: {exc}") from exc

        object_ids = np.unique(mask_array)
        object_ids = object_ids[object_ids != 0]
        if len(object_ids) == 0:
            raise DatasetItemError(f"mask has no object instances: {mask_path}")
        masks = torch.as_tensor(mask_array == object_ids[:, None, None], dtype=torch.uint8)
        boxes: list[list[float]] = []
        for instance_mask in masks:
            positions = torch.where(instance_mask > 0)
            y_min, y_max = positions[0].min().item(), positions[0].max().item() + 1
            x_min, x_max = positions[1].min().item(), positions[1].max().item() + 1
            boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        height, width = image.shape[-2:]
        validate_boxes(box_tensor, width=width, height=height)
        target: Target = {
            "boxes": box_tensor,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": masks,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0])
            * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "source_id": entry.image_id,
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            validate_boxes(target["boxes"], width=image.shape[-1], height=image.shape[-2])
        return image, target


def collate_fn(
    batch: Sequence[tuple[torch.Tensor, Target]],
) -> tuple[tuple[torch.Tensor, ...], tuple[Target, ...]]:
    """Keep variable-size images and targets as tuples for Torchvision."""
    images, targets = zip(*batch, strict=True)
    return images, targets
