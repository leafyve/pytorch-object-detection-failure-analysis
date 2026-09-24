"""Reconstruct a trained detector from public configuration and checkpoint metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from podfa.models.factory import build_model
from podfa.training.checkpoint import load_checkpoint


def load_trained_model(
    model_config: dict[str, Any], checkpoint: Path | str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build without network downloads, validate metadata, load weights, and enter eval mode."""
    architecture = str(model_config["architecture"])
    num_classes = int(model_config["num_classes"])
    model = build_model(
        architecture,
        num_classes,
        False,
        int(model_config["min_size"]),
        int(model_config["max_size"]),
    )
    state = load_checkpoint(
        checkpoint,
        model,
        expected_architecture=architecture,
        expected_num_classes=num_classes,
    )
    model.to(device)
    model.eval()
    return model, state
