"""Reusable detector training loop with validation-only best selection."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from podfa.training.checkpoint import save_checkpoint

Batch = tuple[
    list[torch.Tensor] | tuple[torch.Tensor, ...],
    list[dict[str, Any]] | tuple[dict[str, Any], ...],
]


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_map: float
    duration_seconds: float
    history: tuple[dict[str, Any], ...]


def _to_device_target(target: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    """Train for one epoch and return sample-weighted detector losses."""
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    for images, targets in loader:
        device_images = [image.to(device) for image in images]
        device_targets = [_to_device_target(target, device) for target in targets]
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and scaler.is_enabled()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(device_images, device_targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {float(loss.detach().cpu())}")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        for name, value in loss_dict.items():
            totals[name] += float(value.detach().cpu())
        batches += 1
    if batches == 0:
        raise ValueError("training loader yielded no batches")
    return {name: value / batches for name, value in sorted(totals.items())}


def fit_detector(
    *,
    model: nn.Module,
    train_loader: Iterable[Batch],
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    device: torch.device,
    epochs: int,
    output_dir: Path | str,
    metadata: dict[str, Any],
    validation_callback: Callable[[nn.Module], dict[str, float]],
    scaler: torch.amp.GradScaler | None,
    start_epoch: int = 1,
    initial_best: float = float("-inf"),
    metric_logger: Callable[[dict[str, float], int], None] | None = None,
) -> TrainingResult:
    """Fit a detector and checkpoint only from validation mAP@0.50:0.95."""
    if epochs < start_epoch:
        raise ValueError("epochs must be greater than or equal to start_epoch")
    output_path = Path(output_dir)
    checkpoints = output_path / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = initial_best
    best_epoch = start_epoch - 1
    started = time.perf_counter()
    model.to(device)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler)
        validation_metrics = validation_callback(model)
        if "map_50_95" not in validation_metrics:
            raise ValueError("validation callback must return map_50_95")
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        if metric_logger is not None:
            numeric_record = {
                key: float(value) for key, value in record.items() if key != "epoch"
            }
            metric_logger(numeric_record, epoch)
        score = float(validation_metrics["map_50_95"])
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_validation_map": max(best_score, score),
            "metadata": metadata,
            "history": history,
        }
        save_checkpoint(checkpoints / "latest.pth", state)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            state["best_validation_map"] = best_score
            save_checkpoint(checkpoints / "best.pth", state)
        if scheduler is not None:
            scheduler.step()
        (output_path / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    duration = time.perf_counter() - started
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_map=best_score,
        duration_seconds=duration,
        history=tuple(history),
    )
