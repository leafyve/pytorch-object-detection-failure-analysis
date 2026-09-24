"""Configuration-driven end-to-end detector training pipeline."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from podfa.data.dataset import PennFudanDataset, collate_fn
from podfa.data.transforms import build_transforms
from podfa.evaluation.evaluator import collect_predictions, evaluate_predictions
from podfa.models.factory import PRETRAINED_WEIGHTS_NAME, build_model, trainable_parameter_count
from podfa.training.checkpoint import load_checkpoint
from podfa.training.mlflow_tracking import LocalMLflowTracker
from podfa.training.trainer import fit_detector
from podfa.utils.config import load_config
from podfa.utils.environment import collect_environment, select_device
from podfa.utils.reproducibility import seed_everything


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested configuration values into stable MLflow parameter keys."""
    flattened: dict[str, Any] = {}
    for key in sorted(config):
        value = config[key]
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_config(value, full_key))
        elif isinstance(value, list):
            flattened[full_key] = ",".join(str(item) for item in value)
        else:
            flattened[full_key] = value
    return flattened


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> Optimizer:
    """Create SGD over trainable parameters only."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_training(config_path: Path | str, resume: Path | str | None = None) -> dict[str, Any]:
    """Train one configured experiment and return its evidence summary."""
    config_file = Path(config_path)
    config = load_config(config_file)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = select_device()
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "history.json").exists() and resume is None:
        raise FileExistsError(
            f"experiment output already contains history: {output_dir}; use --resume or a new name"
        )

    environment = collect_environment()
    environment_path = output_dir / "environment.json"
    _write_json(environment_path, environment)
    config_snapshot = output_dir / "config.yaml"
    shutil.copy2(config_file, config_snapshot)

    experiment_name = str(config["experiment"]["name"])
    train_dataset = PennFudanDataset(
        config["data"]["root"],
        config["data"]["manifest"],
        "train",
        build_transforms(True, experiment_name),
    )
    validation_dataset = PennFudanDataset(
        config["data"]["root"],
        config["data"]["manifest"],
        "validation",
        build_transforms(False, experiment_name),
    )
    generator = torch.Generator().manual_seed(seed)
    num_workers = int(config["training"].get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
    )

    model_config = config["model"]
    model = build_model(
        model_config["architecture"],
        int(model_config["num_classes"]),
        bool(model_config["pretrained"]),
        int(model_config["min_size"]),
        int(model_config["max_size"]),
    )
    training_config = config["training"]
    optimizer = build_optimizer(
        model,
        float(training_config["learning_rate"]),
        float(training_config.get("momentum", 0.9)),
        float(training_config.get("weight_decay", 0.0005)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training_config.get("scheduler_step_size", 3)),
        gamma=float(training_config.get("scheduler_gamma", 0.1)),
    )
    use_amp = bool(training_config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    initial_best = float("-inf")
    if resume is not None:
        state = load_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            expected_architecture=model_config["architecture"],
            expected_num_classes=int(model_config["num_classes"]),
        )
        start_epoch = int(state["epoch"]) + 1
        initial_best = float(state.get("best_validation_map", float("-inf")))
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])

    metadata = {
        "architecture": model_config["architecture"],
        "num_classes": int(model_config["num_classes"]),
        "pretrained_weights": PRETRAINED_WEIGHTS_NAME if model_config["pretrained"] else "none",
        "seed": seed,
        "config_sha256": _sha256(config_snapshot),
        "manifest_sha256": _sha256(Path(config["data"]["manifest"])),
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
    }

    with LocalMLflowTracker(Path("mlruns"), "podfa-penn-fudan", experiment_name) as tracker:
        metadata["mlflow_run_id"] = tracker.run_id
        tracker.log_params({**flatten_config(config), **metadata})
        tracker.log_artifact(config_snapshot, "run")
        tracker.log_artifact(environment_path, "run")

        def validation_callback(current_model: nn.Module) -> dict[str, float]:
            predictions, targets = collect_predictions(current_model, validation_loader, device)
            metrics = evaluate_predictions(
                predictions,
                targets,
                float(config["evaluation"]["confidence_threshold"]),
                float(config["evaluation"]["iou_threshold"]),
            )
            return {key: float(value) for key, value in metrics.items()}

        result = fit_detector(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=int(training_config["epochs"]),
            output_dir=output_dir,
            metadata=metadata,
            validation_callback=validation_callback,
            scaler=scaler,
            start_epoch=start_epoch,
            initial_best=initial_best,
            metric_logger=tracker.log_metrics,
        )
        summary = {
            **metadata,
            "best_epoch": result.best_epoch,
            "best_validation_map_50_95": result.best_validation_map,
            "training_duration_seconds": result.duration_seconds,
            "completed_epochs": len(result.history),
        }
        _write_json(output_dir / "run_summary.json", summary)
        tracker.log_metrics(
            {
                "best_validation_map_50_95": result.best_validation_map,
                "training_duration_seconds": result.duration_seconds,
            }
        )
        tracker.log_artifact(output_dir / "history.json", "run")
        tracker.log_artifact(output_dir / "run_summary.json", "run")
    return summary
