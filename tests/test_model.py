from __future__ import annotations

import pytest
import torch

from podfa.models.factory import ModelFactoryError, build_model


def test_build_model_replaces_classifier_for_background_and_person() -> None:
    model = build_model(
        "fasterrcnn_mobilenet_v3_large_fpn",
        num_classes=2,
        pretrained=False,
        min_size=64,
        max_size=96,
    )

    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.transform.min_size == (64,)
    assert model.transform.max_size == 96


def test_build_model_rejects_unknown_architecture() -> None:
    with pytest.raises(ModelFactoryError, match="unsupported architecture"):
        build_model("not-a-model", 2, False, 64, 96)


@pytest.mark.model_smoke
def test_real_detector_runs_inference_and_training_forward() -> None:
    model = build_model(
        "fasterrcnn_mobilenet_v3_large_fpn", 2, False, min_size=64, max_size=96
    )
    images = [torch.rand(3, 64, 64), torch.rand(3, 72, 64)]
    targets = [
        {
            "boxes": torch.tensor([[8.0, 8.0, 40.0, 52.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[6.0, 10.0, 42.0, 60.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
    ]

    model.eval()
    with torch.inference_mode():
        predictions = model(images)
    model.train()
    losses = model(images, targets)

    assert len(predictions) == 2
    assert {"boxes", "labels", "scores"} <= predictions[0].keys()
    assert {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"} <= (
        losses.keys()
    )
    assert all(torch.isfinite(value) for value in losses.values())
