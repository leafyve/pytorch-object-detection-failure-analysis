"""Safe single-image inference with atomic annotated output."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision.transforms.functional import pil_to_tensor

from podfa.evaluation.visualize import render_detection


class ImageInferenceError(RuntimeError):
    """Raised when an input or output prevents complete image inference."""


def predict_image(
    model: nn.Module,
    input_path: Path | str,
    output_path: Path | str,
    confidence_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    """Run one image through a detector and atomically save its overlay."""
    source = Path(input_path)
    output = Path(output_path)
    try:
        with Image.open(source) as opened:
            pil_image = opened.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageInferenceError(f"unreadable image {source}: {exc}") from exc
    image_tensor = pil_to_tensor(pil_image).float().div(255.0)
    model.to(device)
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model([image_tensor.to(device)])[0]
    duration = time.perf_counter() - started
    prediction = {key: value.detach().cpu() for key, value in prediction.items()}
    rendered = render_detection(pil_image, None, prediction, confidence_threshold)
    selected = (prediction["scores"] >= confidence_threshold) & (prediction["labels"] == 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    try:
        rendered.save(temporary)
        os.replace(temporary, output)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ImageInferenceError(f"could not write annotated image {output}: {exc}") from exc
    return {
        "input": str(source),
        "output": str(output),
        "image_size": [pil_image.width, pil_image.height],
        "detections": int(selected.sum().item()),
        "confidence_threshold": confidence_threshold,
        "inference_seconds": duration,
        "device": str(device),
    }
