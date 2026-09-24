from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from podfa.inference.video import VideoInferenceError, predict_video


class FixedVideoDetector(torch.nn.Module):
    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor([[2.0, 2.0, 14.0, 16.0]], device=images[0].device),
                "scores": torch.tensor([0.9], device=images[0].device),
                "labels": torch.tensor([1], device=images[0].device),
            }
            for _ in images
        ]


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (24, 20)
    )
    assert writer.isOpened()
    writer.write(np.zeros((20, 24, 3), dtype=np.uint8))
    writer.write(np.full((20, 24, 3), 40, dtype=np.uint8))
    writer.release()


def test_predict_video_annotates_every_frame_and_reports_fps(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    output = tmp_path / "annotated.avi"
    _write_video(source)

    summary = predict_video(
        FixedVideoDetector(),
        source,
        output,
        confidence_threshold=0.5,
        device=torch.device("cpu"),
    )

    capture = cv2.VideoCapture(str(output))
    written_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert summary.frame_count == 2
    assert written_frames == 2
    assert summary.average_inference_fps > 0


def test_predict_video_rejects_corrupt_input_without_partial_output(tmp_path: Path) -> None:
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"not a video")
    output = tmp_path / "annotated.mp4"

    with pytest.raises(VideoInferenceError, match="unreadable video"):
        predict_video(
            FixedVideoDetector(),
            source,
            output,
            confidence_threshold=0.5,
            device=torch.device("cpu"),
        )

    assert not output.exists()
