from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from podfa.training.trainer import fit_detector, train_one_epoch


class TinyDetector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, Any]] | None = None
    ) -> dict[str, torch.Tensor]:
        assert targets is not None
        target_value = targets[0]["value"]
        return {"loss_classifier": (self.weight * images[0].mean() - target_value) ** 2}


def _loader() -> list[tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]]:
    return [([torch.ones(1, 2, 2)], [{"value": torch.tensor(0.0)}])]


def test_train_one_epoch_updates_parameter_and_reports_mean_loss() -> None:
    model = TinyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_one_epoch(model, _loader(), optimizer, torch.device("cpu"), scaler=None)

    assert metrics["loss"] == 1.0
    assert metrics["loss_classifier"] == 1.0
    assert model.weight.item() < 1.0


def test_fit_detector_selects_best_checkpoint_only_by_validation_score(tmp_path: Path) -> None:
    model = TinyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scores = iter([0.8, 0.2])

    result = fit_detector(
        model=model,
        train_loader=_loader(),
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        epochs=2,
        output_dir=tmp_path,
        metadata={"architecture": "fixture", "num_classes": 2},
        validation_callback=lambda _model: {"map_50_95": next(scores)},
        scaler=None,
    )

    assert result.best_epoch == 1
    assert result.best_validation_map == 0.8
    assert len(result.history) == 2
    assert (tmp_path / "checkpoints" / "best.pth").is_file()
    best = torch.load(tmp_path / "checkpoints" / "best.pth", weights_only=False)
    assert best["epoch"] == 1
