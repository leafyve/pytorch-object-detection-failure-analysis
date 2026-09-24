from __future__ import annotations

from pathlib import Path

import pytest
import torch

from podfa.training.checkpoint import CheckpointError, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    with torch.no_grad():
        model.weight.fill_(3.0)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 4,
        "best_validation_map": 0.42,
        "metadata": {"architecture": "fixture", "num_classes": 2},
    }
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, state)
    with torch.no_grad():
        model.weight.zero_()

    restored = load_checkpoint(path, model, optimizer, expected_architecture="fixture")

    assert model.weight.tolist() == [[3.0, 3.0]]
    assert restored["epoch"] == 4
    assert not path.with_suffix(".pth.tmp").exists()


def test_checkpoint_rejects_architecture_mismatch_before_loading(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(
        path,
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "metadata": {"architecture": "fixture", "num_classes": 2},
        },
    )

    with pytest.raises(CheckpointError, match="architecture mismatch"):
        load_checkpoint(path, model, expected_architecture="different")


def test_checkpoint_rejects_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(b"not a torch checkpoint")

    with pytest.raises(CheckpointError, match="could not load"):
        load_checkpoint(path, torch.nn.Linear(2, 1))
