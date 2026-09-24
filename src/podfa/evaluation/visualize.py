"""Detection overlays, experiment plots, and failure galleries."""

from __future__ import annotations

import csv
import html
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from podfa.evaluation.failure_analysis import FailureRecord, categorize_failures


def _to_pil(image: torch.Tensor | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def render_detection(
    image: torch.Tensor | Image.Image,
    target: dict[str, Any] | None,
    prediction: dict[str, torch.Tensor],
    confidence_threshold: float,
) -> Image.Image:
    """Overlay ground truth in green and selected predictions in red."""
    canvas = _to_pil(image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if target is not None:
        for box in target["boxes"].detach().cpu().tolist():
            draw.rectangle(box, outline=(46, 204, 113), width=2)
            draw.text((box[0] + 2, box[1] + 2), "GT", fill=(46, 204, 113), font=font)
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    for box, score, label in zip(boxes, scores, labels, strict=True):
        if int(label.item()) != 1 or float(score.item()) < confidence_threshold:
            continue
        coordinates = box.tolist()
        draw.rectangle(coordinates, outline=(231, 76, 60), width=2)
        draw.text(
            (coordinates[0] + 2, max(0, coordinates[1] - 10)),
            f"person {float(score.item()):.2f}",
            fill=(231, 76, 60),
            font=font,
        )
    return canvas


def write_failure_csv(records: list[FailureRecord], path: Path | str) -> None:
    """Write structured failure evidence with stable columns."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "failure_type",
        "confidence",
        "iou",
        "ground_truth_count",
        "prediction_count",
        "prediction_index",
        "ground_truth_index",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)


def analyze_prediction_bundle(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
    confidence_threshold: float,
) -> tuple[list[FailureRecord], dict[str, int]]:
    """Categorize every image and summarize actual failure counts."""
    records = [
        failure
        for prediction, target in zip(predictions, targets, strict=True)
        for failure in categorize_failures(prediction, target, confidence_threshold)
    ]
    return records, dict(sorted(Counter(record.failure_type for record in records).items()))


def plot_training_history(history: list[dict[str, Any]], output_dir: Path | str) -> None:
    """Write training-loss and validation-metric plots from epoch history."""
    if not history:
        raise ValueError("training history is empty")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [record["train_loss"] for record in history], marker="o")
    axis.set(xlabel="Epoch", ylabel="Detector loss", title="Training loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination / "training_loss.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (
        ("validation_map_50_95", "mAP@0.50:0.95"),
        ("validation_map_50", "mAP@0.50"),
        ("validation_precision", "Precision @ 0.50"),
        ("validation_recall", "Recall @ 0.50"),
    ):
        if key in history[0]:
            axis.plot(epochs, [record[key] for record in history], marker="o", label=label)
    axis.set(xlabel="Epoch", ylabel="Metric", title="Validation metrics", ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "validation_metrics.png", dpi=160)
    plt.close(figure)


def write_detection_gallery(
    dataset: Any,
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
    confidence_threshold: float,
    output_dir: Path | str,
    limit: int = 8,
) -> None:
    """Save representative annotated predictions from an evaluation split."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        if index >= limit:
            break
        image, _ = dataset[index]
        rendered = render_detection(image, target, prediction, confidence_threshold)
        rendered.save(destination / f"{target['source_id']}.jpg", quality=90)


def write_failure_gallery(
    dataset: Any,
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
    failures: list[FailureRecord],
    confidence_threshold: float,
    output_dir: Path | str,
    limit: int = 16,
) -> None:
    """Save one annotated panel per image with actual failure labels."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[FailureRecord]] = {}
    for failure in failures:
        by_image.setdefault(failure.image_id, []).append(failure)
    cards: list[str] = []
    written = 0
    for index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        source_id = str(target["source_id"])
        if source_id not in by_image or written >= limit:
            continue
        image, _ = dataset[index]
        rendered = render_detection(image, target, prediction, confidence_threshold)
        filename = f"{source_id}.jpg"
        rendered.save(destination / filename, quality=90)
        labels = ", ".join(sorted({item.failure_type for item in by_image[source_id]}))
        cards.append(
            f'<figure><img src="{html.escape(filename)}" alt="Detection failures for '
            f'{html.escape(source_id)}"><figcaption><strong>{html.escape(source_id)}</strong>: '
            f"{html.escape(labels)}</figcaption></figure>"
        )
        written += 1
    page = (
        "<!doctype html><meta charset=\"utf-8\"><title>Validation failure gallery</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}"
        "figure{margin:0;border:1px solid #ddd;padding:.75rem;border-radius:.5rem}"
        "img{max-width:100%;height:auto}figcaption{margin-top:.5rem}</style>"
        "<h1>Validation failure gallery</h1><main>"
        + "".join(cards)
        + "</main>"
    )
    (destination / "index.html").write_text(page, encoding="utf-8")
