"""Download-free installation and model inference smoke test."""

from __future__ import annotations

import json

import torch

from podfa.models.factory import build_model
from podfa.utils.config import load_config


def main() -> int:
    config = load_config("configs/smoke.yaml")
    model_config = config["model"]
    model = build_model(
        model_config["architecture"],
        model_config["num_classes"],
        False,
        model_config["min_size"],
        model_config["max_size"],
    )
    model.eval()
    with torch.inference_mode():
        prediction = model([torch.zeros(3, 128, 128)])[0]
    payload = {
        "package_import": "ok",
        "config_parse": "ok",
        "model_architecture": model_config["architecture"],
        "model_inference": "ok",
        "prediction_fields": sorted(prediction),
        "device": "cpu",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
