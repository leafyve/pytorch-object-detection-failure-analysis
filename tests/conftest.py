from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from podfa.data.splits import SplitEntry, SplitManifest


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> tuple[Path, SplitManifest]:
    root = tmp_path / "PennFudanPed"
    (root / "PNGImages").mkdir(parents=True)
    (root / "PedMasks").mkdir()
    image = Image.new("RGB", (10, 8), (20, 40, 60))
    image.save(root / "PNGImages" / "sample.png")
    mask = torch.zeros((8, 10), dtype=torch.uint8)
    mask[1:7, 2:6] = 1
    Image.fromarray(mask.numpy()).save(root / "PedMasks" / "sample_mask.png")
    entry = SplitEntry(
        image_id="sample",
        image_file="sample.png",
        mask_file="sample_mask.png",
        sha256="a" * 64,
    )
    manifest = SplitManifest(
        seed=42,
        ratios={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
        partitions={"train": (entry,), "validation": (entry,), "test": (entry,)},
    )
    return root, manifest
