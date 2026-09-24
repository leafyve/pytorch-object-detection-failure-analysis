from __future__ import annotations

import torch

from podfa.training.pipeline import build_optimizer, flatten_config


def test_flatten_config_produces_stable_mlflow_parameter_names() -> None:
    config = {
        "model": {"architecture": "fixture", "pretrained": True},
        "training": {"epochs": 2, "thresholds": [0.25, 0.5]},
    }

    assert flatten_config(config) == {
        "model.architecture": "fixture",
        "model.pretrained": True,
        "training.epochs": 2,
        "training.thresholds": "0.25,0.5",
    }


def test_build_optimizer_excludes_frozen_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    optimizer = build_optimizer(model, learning_rate=0.01, momentum=0.9, weight_decay=0.001)

    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model[1].parameters()}
    assert optimizer.param_groups[0]["lr"] == 0.01
