from __future__ import annotations

from pathlib import Path

import torch

from podfa.models.factory import build_model
from podfa.models.loading import load_trained_model
from podfa.training.checkpoint import save_checkpoint


def test_load_trained_model_reconstructs_architecture_without_pretrained_download(
    tmp_path: Path,
) -> None:
    config = {
        "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
        "num_classes": 2,
        "min_size": 64,
        "max_size": 96,
    }
    source = build_model(config["architecture"], 2, False, 64, 96)
    checkpoint = tmp_path / "model.pth"
    save_checkpoint(
        checkpoint,
        {
            "model_state_dict": source.state_dict(),
            "epoch": 1,
            "metadata": {"architecture": config["architecture"], "num_classes": 2},
        },
    )

    loaded, state = load_trained_model(config, checkpoint, torch.device("cpu"))

    assert loaded.training is False
    assert state["epoch"] == 1
    assert loaded.roi_heads.box_predictor.cls_score.out_features == 2
