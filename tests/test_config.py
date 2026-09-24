from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from podfa.utils.config import ConfigError, load_config
from podfa.utils.environment import collect_environment, select_device
from podfa.utils.reproducibility import seed_everything


def _valid_config() -> dict[str, object]:
    return {
        "experiment": {"name": "smoke", "seed": 42, "output_dir": "artifacts/smoke"},
        "data": {"root": "data/raw/PennFudanPed", "manifest": "artifacts/splits/seed42.json"},
        "model": {
            "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
            "pretrained": False,
            "num_classes": 2,
            "min_size": 128,
            "max_size": 256,
        },
        "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.001},
        "evaluation": {"confidence_threshold": 0.5, "iou_threshold": 0.5},
    }


def test_load_config_accepts_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_valid_config()), encoding="utf-8")

    loaded = load_config(path)

    assert loaded["model"]["architecture"] == "fasterrcnn_mobilenet_v3_large_fpn"
    assert loaded["experiment"]["seed"] == 42


def test_load_config_rejects_missing_required_section(tmp_path: Path) -> None:
    config = _valid_config()
    del config["training"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="training"):
        load_config(path)


def test_load_config_rejects_unknown_architecture(tmp_path: Path) -> None:
    config = _valid_config()
    config["model"]["architecture"] = "imaginary_detector"  # type: ignore[index]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="architecture"):
        load_config(path)


def test_seed_everything_repeats_python_numpy_and_torch() -> None:
    seed_everything(7)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    seed_everything(7)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    assert second == first


def test_select_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert select_device().type == "cpu"


def test_collect_environment_contains_runtime_fields() -> None:
    metadata = collect_environment()

    assert metadata["python_version"]
    assert metadata["platform"]
    assert metadata["torch_version"]
    assert metadata["device"]["type"] in {"cpu", "cuda", "mps"}
