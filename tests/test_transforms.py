from __future__ import annotations

import torch

from podfa.data.transforms import DetectionTransform, build_transforms


def test_horizontal_flip_updates_boxes_without_mutating_target() -> None:
    image = torch.arange(3 * 4 * 10, dtype=torch.float32).reshape(3, 4, 10)
    target = {"boxes": torch.tensor([[2.0, 1.0, 6.0, 3.0]]), "labels": torch.tensor([1])}
    original_boxes = target["boxes"].clone()
    transform = DetectionTransform(training=True, horizontal_flip_probability=1.0)

    flipped_image, flipped_target = transform(image, target)

    assert torch.equal(flipped_image, torch.flip(image, dims=(-1,)))
    assert flipped_target["boxes"].tolist() == [[4.0, 1.0, 8.0, 3.0]]
    assert torch.equal(target["boxes"], original_boxes)


def test_evaluation_transform_is_identity_for_tensor_and_boxes() -> None:
    image = torch.rand(3, 6, 8)
    target = {"boxes": torch.tensor([[1.0, 1.0, 4.0, 5.0]]), "labels": torch.tensor([1])}

    transformed_image, transformed_target = build_transforms(False, "baseline")(image, target)

    assert torch.equal(transformed_image, image)
    assert torch.equal(transformed_target["boxes"], target["boxes"])


def test_transform_rejects_unknown_experiment() -> None:
    try:
        build_transforms(True, "unknown")
    except ValueError as exc:
        assert "unknown transform experiment" in str(exc)
    else:
        raise AssertionError("unknown experiment was accepted")
