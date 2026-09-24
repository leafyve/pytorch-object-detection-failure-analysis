"""Strict YAML configuration loading for repeatable experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SUPPORTED_ARCHITECTURES = {"fasterrcnn_mobilenet_v3_large_fpn"}
REQUIRED_SECTIONS = {"experiment", "data", "model", "training", "evaluation"}


class ConfigError(ValueError):
    """Raised when an experiment configuration is missing or malformed."""


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def load_config(path: Path | str) -> dict[str, Any]:
    """Load and validate the stable public configuration contract."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    config = _require_mapping(raw, "configuration")
    missing = sorted(REQUIRED_SECTIONS - config.keys())
    if missing:
        raise ConfigError(f"missing required configuration sections: {', '.join(missing)}")
    for section in REQUIRED_SECTIONS:
        _require_mapping(config[section], section)

    model = config["model"]
    architecture = model.get("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ConfigError(
            f"unsupported model architecture {architecture!r}; "
            f"choose one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    if model.get("num_classes") != 2:
        raise ConfigError("model.num_classes must be 2 (background and person)")
    for field in ("epochs", "batch_size"):
        value = config["training"].get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"training.{field} must be a positive integer")
    learning_rate = config["training"].get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ConfigError("training.learning_rate must be positive")
    seed = config["experiment"].get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("experiment.seed must be a non-negative integer")
    for field in ("confidence_threshold", "iou_threshold"):
        value = config["evaluation"].get(field)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ConfigError(f"evaluation.{field} must be between 0 and 1")
    return config
