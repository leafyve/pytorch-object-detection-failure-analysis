from __future__ import annotations

from pathlib import Path

import pytest
import torch

from podfa.data.dataset import BoundingBoxError, PennFudanDataset, collate_fn, validate_boxes
from podfa.data.splits import SplitManifest


def test_dataset_loads_image_and_torchvision_target(
    synthetic_dataset: tuple[Path, SplitManifest],
) -> None:
    root, manifest = synthetic_dataset
    dataset = PennFudanDataset(root, manifest, "train")

    image, target = dataset[0]

    assert image.shape == (3, 8, 10)
    assert image.dtype == torch.float32
    assert target["boxes"].tolist() == [[2.0, 1.0, 6.0, 7.0]]
    assert target["labels"].tolist() == [1]
    assert target["area"].tolist() == [24.0]
    assert target["iscrowd"].tolist() == [0]
    assert target["source_id"] == "sample"


@pytest.mark.parametrize(
    ("boxes", "message"),
    [
        (torch.tensor([[2.0, 1.0, 2.0, 4.0]]), "positive area"),
        (torch.tensor([[-1.0, 1.0, 2.0, 4.0]]), "image bounds"),
        (torch.tensor([[1.0, 1.0, 11.0, 4.0]]), "image bounds"),
        (torch.tensor([[float("nan"), 1.0, 2.0, 4.0]]), "finite"),
        (torch.tensor([1.0, 2.0, 3.0, 4.0]), "shape"),
    ],
)
def test_validate_boxes_rejects_malformed_boxes(boxes: torch.Tensor, message: str) -> None:
    with pytest.raises(BoundingBoxError, match=message):
        validate_boxes(boxes, width=10, height=8)


def test_collate_fn_preserves_variable_targets(
    synthetic_dataset: tuple[Path, SplitManifest],
) -> None:
    root, manifest = synthetic_dataset
    sample = PennFudanDataset(root, manifest, "train")[0]

    images, targets = collate_fn([sample, sample])

    assert len(images) == 2
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
