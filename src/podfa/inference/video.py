"""Frame-by-frame OpenCV inference with annotated atomic output."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from torch import nn


class VideoInferenceError(RuntimeError):
    """Raised when a video cannot be read, processed, or fully written."""


@dataclass(frozen=True)
class VideoSummary:
    input_path: str
    output_path: str
    frame_count: int
    source_fps: float
    average_inference_fps: float
    confidence_threshold: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _codec_for(path: Path) -> int:
    return cv2.VideoWriter_fourcc(*("mp4v" if path.suffix.lower() == ".mp4" else "MJPG"))


def predict_video(
    model: nn.Module,
    input_path: Path | str,
    output_path: Path | str,
    confidence_threshold: float,
    device: torch.device,
) -> VideoSummary:
    """Annotate every readable frame and replace output only after full completion."""
    source = Path(input_path)
    output = Path(output_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise VideoInferenceError(f"unreadable video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise VideoInferenceError(f"video has invalid dimensions: {source}")
    output_fps = source_fps if source_fps > 0 else 25.0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    writer = cv2.VideoWriter(
        str(temporary), _codec_for(output), output_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise VideoInferenceError(f"could not initialize video writer: {output}")

    model.to(device)
    model.eval()
    frame_count = 0
    inference_seconds = 0.0
    try:
        while True:
            readable, frame = capture.read()
            if not readable:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = model([tensor])[0]
            elapsed = time.perf_counter() - started
            inference_seconds += elapsed
            boxes = prediction["boxes"].detach().cpu()
            scores = prediction["scores"].detach().cpu()
            labels = prediction["labels"].detach().cpu()
            for box, score, label in zip(boxes, scores, labels, strict=True):
                score_value = float(score.item())
                if int(label.item()) != 1 or score_value < confidence_threshold:
                    continue
                x_min, y_min, x_max, y_max = (int(value) for value in box.tolist())
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 210, 255), 2)
                cv2.putText(
                    frame,
                    f"person {score_value:.2f}",
                    (x_min, max(15, y_min - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 210, 255),
                    1,
                    cv2.LINE_AA,
                )
            instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame,
                f"Inference FPS: {instantaneous_fps:.1f}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 255, 80),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_count += 1
    except Exception as exc:
        raise VideoInferenceError(f"video inference failed at frame {frame_count}: {exc}") from exc
    finally:
        capture.release()
        writer.release()
    if frame_count == 0:
        temporary.unlink(missing_ok=True)
        raise VideoInferenceError(f"unreadable video: no frames decoded from {source}")
    try:
        os.replace(temporary, output)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise VideoInferenceError(f"could not finalize annotated video {output}: {exc}") from exc
    return VideoSummary(
        input_path=str(source),
        output_path=str(output),
        frame_count=frame_count,
        source_fps=source_fps,
        average_inference_fps=frame_count / inference_seconds if inference_seconds else 0.0,
        confidence_threshold=confidence_threshold,
        device=str(device),
    )
