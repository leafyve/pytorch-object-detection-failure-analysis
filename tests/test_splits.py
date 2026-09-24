from __future__ import annotations

import json
from pathlib import Path

import pytest

from podfa.data.splits import (
    SplitValidationError,
    create_split_manifest,
    save_split_manifest,
    validate_split_manifest,
)
from podfa.data.validation import DatasetRecord


def _records(count: int = 170) -> list[DatasetRecord]:
    return [
        DatasetRecord(
            image_id=f"image-{index:03d}",
            image_path=Path(f"PNGImages/image-{index:03d}.png"),
            mask_path=Path(f"PedMasks/image-{index:03d}_mask.png"),
            sha256=f"{index:064x}",
        )
        for index in range(count)
    ]


def test_seed_42_manifest_is_byte_reproducible(tmp_path: Path) -> None:
    first = create_split_manifest(_records(), seed=42, ratios=(0.70, 0.15, 0.15))
    second = create_split_manifest(_records(), seed=42, ratios=(0.70, 0.15, 0.15))
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    save_split_manifest(first, first_path)
    save_split_manifest(second, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.counts == {"train": 119, "validation": 25, "test": 26}


def test_different_seed_changes_split_membership() -> None:
    first = create_split_manifest(_records(), seed=42, ratios=(0.70, 0.15, 0.15))
    second = create_split_manifest(_records(), seed=43, ratios=(0.70, 0.15, 0.15))

    assert first.partitions["train"] != second.partitions["train"]


def test_validate_manifest_rejects_duplicate_id_across_splits() -> None:
    manifest = create_split_manifest(_records(), seed=42, ratios=(0.70, 0.15, 0.15))
    duplicate = manifest.partitions["train"][0]
    manifest.partitions["test"] = (duplicate, *manifest.partitions["test"][1:])

    with pytest.raises(SplitValidationError, match="image ID overlap"):
        validate_split_manifest(manifest)


def test_validate_manifest_rejects_duplicate_hash_across_splits() -> None:
    manifest = create_split_manifest(_records(), seed=42, ratios=(0.70, 0.15, 0.15))
    train_entry = manifest.partitions["train"][0]
    test_entry = manifest.partitions["test"][0]
    manifest.partitions["test"] = (
        test_entry.__class__(
            image_id=test_entry.image_id,
            image_file=test_entry.image_file,
            mask_file=test_entry.mask_file,
            sha256=train_entry.sha256,
        ),
        *manifest.partitions["test"][1:],
    )

    with pytest.raises(SplitValidationError, match="content hash overlap"):
        validate_split_manifest(manifest)


def test_saved_manifest_contains_seed_ratios_and_all_ids(tmp_path: Path) -> None:
    manifest = create_split_manifest(_records(), seed=42, ratios=(0.70, 0.15, 0.15))
    path = tmp_path / "manifest.json"
    save_split_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["seed"] == 42
    assert payload["ratios"] == {"train": 0.7, "validation": 0.15, "test": 0.15}
    assert sum(len(items) for items in payload["partitions"].values()) == 170
