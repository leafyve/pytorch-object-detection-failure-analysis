"""Checkpoint evaluation workflow shared by validation and held-out test CLIs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from podfa.data.dataset import PennFudanDataset, collate_fn
from podfa.data.transforms import build_transforms
from podfa.evaluation.evaluator import (
    collect_predictions,
    evaluate_predictions,
    save_prediction_bundle,
    select_confidence_threshold,
    write_metrics,
)
from podfa.evaluation.visualize import (
    analyze_prediction_bundle,
    plot_training_history,
    write_detection_gallery,
    write_failure_csv,
    write_failure_gallery,
)
from podfa.models.loading import load_trained_model
from podfa.utils.config import load_config
from podfa.utils.environment import select_device
from podfa.utils.reproducibility import seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_evaluation(
    *,
    config_path: Path | str,
    checkpoint_path: Path | str,
    split: str,
    output_dir: Path | str,
    confidence_threshold: float | None = None,
    select_threshold: bool = False,
    failure_csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable checkpoint on a named manifest partition."""
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if split == "test" and select_threshold:
        raise ValueError("threshold selection on the held-out test split is forbidden")
    config = load_config(config_path)
    seed_everything(int(config["experiment"]["seed"]))
    device = select_device()
    dataset = PennFudanDataset(
        config["data"]["root"],
        config["data"]["manifest"],
        split,
        build_transforms(False, str(config["experiment"]["name"])),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    checkpoint = Path(checkpoint_path)
    model, state = load_trained_model(config["model"], checkpoint, device)
    started = time.perf_counter()
    predictions, targets = collect_predictions(model, loader, device)
    inference_duration = time.perf_counter() - started
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_prediction_bundle(predictions, targets, output / "predictions.pt")

    threshold_selection: dict[str, Any] | None = None
    if select_threshold:
        candidates = [float(value) for value in config["evaluation"]["threshold_candidates"]]
        selection = select_confidence_threshold(
            predictions,
            targets,
            candidates,
            float(config["evaluation"]["iou_threshold"]),
        )
        threshold = selection.threshold
        threshold_selection = {
            "selection_split": "validation",
            "objective": "maximum F1; ties resolved by recall, precision, then threshold",
            "selected_threshold": threshold,
            "curve": list(selection.curve),
        }
        write_metrics(threshold_selection, output / "threshold_selection.json")
    elif confidence_threshold is not None:
        threshold = confidence_threshold
    else:
        threshold = float(config["evaluation"]["confidence_threshold"])

    metrics = evaluate_predictions(
        predictions,
        targets,
        threshold,
        float(config["evaluation"]["iou_threshold"]),
    )
    metrics.update(
        {
            "split": split,
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_epoch": int(state["epoch"]),
            "manifest_sha256": state.get("metadata", {}).get("manifest_sha256"),
            "device": str(device),
            "evaluation_duration_seconds": inference_duration,
        }
    )
    write_metrics(metrics, output / "metrics.json")
    write_detection_gallery(dataset, predictions, targets, threshold, output / "samples")
    failures, summary = analyze_prediction_bundle(predictions, targets, threshold)
    write_failure_csv(failures, output / "failures.csv")
    write_metrics(
        {"split": split, "confidence_threshold": threshold, "counts": summary},
        output / "failure_summary.json",
    )
    write_failure_gallery(
        dataset,
        predictions,
        targets,
        failures,
        threshold,
        output / "failure_gallery",
    )
    if failure_csv_path is not None:
        write_failure_csv(failures, failure_csv_path)
    history_path = checkpoint.parent.parent / "history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        plot_training_history(history, checkpoint.parent.parent / "plots")
    return {"metrics": metrics, "failure_counts": summary, "threshold": threshold}
