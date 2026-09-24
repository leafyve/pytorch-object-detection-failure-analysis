"""Atomic, compatibility-checked PyTorch checkpoints."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is partial, malformed, or incompatible."""


def save_checkpoint(path: Path | str, state: dict[str, Any]) -> None:
    """Serialize to a sibling temporary file and atomically replace the target."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise CheckpointError(f"could not save checkpoint {checkpoint_path}: {exc}") from exc


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    expected_architecture: str | None = None,
    expected_num_classes: int | None = None,
) -> dict[str, Any]:
    """Load trusted local state after validating architecture metadata."""
    checkpoint_path = Path(path)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"could not load checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise CheckpointError(f"could not load checkpoint {checkpoint_path}: missing model state")
    metadata = state.get("metadata", {})
    architecture = metadata.get("architecture")
    if expected_architecture is not None and architecture != expected_architecture:
        raise CheckpointError(
            "architecture mismatch: "
            f"expected {expected_architecture}, checkpoint has {architecture}"
        )
    num_classes = metadata.get("num_classes")
    if expected_num_classes is not None and num_classes != expected_num_classes:
        raise CheckpointError(
            f"class-count mismatch: expected {expected_num_classes}, checkpoint has {num_classes}"
        )
    try:
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
    except (RuntimeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"checkpoint state is incompatible: {exc}") from exc
    return state
