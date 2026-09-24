from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from podfa.inference.image import ImageInferenceError, predict_image


class FixedDetector(torch.nn.Module):
    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor([[2.0, 2.0, 12.0, 14.0]], device=images[0].device),
                "scores": torch.tensor([0.9], device=images[0].device),
                "labels": torch.tensor([1], device=images[0].device),
            }
        ]


def test_predict_image_writes_annotation_and_returns_count(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "black").save(source)
    output = tmp_path / "annotated.png"

    summary = predict_image(
        FixedDetector(), source, output, confidence_threshold=0.5, device=torch.device("cpu")
    )

    assert output.is_file()
    assert summary["detections"] == 1
    assert summary["image_size"] == [20, 20]


def test_predict_image_rejects_unreadable_input_without_partial_output(tmp_path: Path) -> None:
    source = tmp_path / "broken.png"
    source.write_bytes(b"not an image")
    output = tmp_path / "annotated.png"

    with pytest.raises(ImageInferenceError, match="unreadable image"):
        predict_image(
            FixedDetector(), source, output, confidence_threshold=0.5, device=torch.device("cpu")
        )

    assert not output.exists()
