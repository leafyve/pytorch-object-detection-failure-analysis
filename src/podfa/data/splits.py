"""Deterministic, leakage-checked train/validation/test manifests."""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from podfa.data.validation import DatasetRecord

SPLIT_NAMES = ("train", "validation", "test")


class SplitValidationError(ValueError):
    """Raised when a manifest leaks identities or has invalid allocation."""


@dataclass(frozen=True)
class SplitEntry:
    image_id: str
    image_file: str
    mask_file: str
    sha256: str


@dataclass
class SplitManifest:
    seed: int
    ratios: dict[str, float]
    partitions: dict[str, tuple[SplitEntry, ...]]

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(self.partitions[name]) for name in SPLIT_NAMES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "ratios": self.ratios,
            "counts": self.counts,
            "partitions": {
                name: [asdict(entry) for entry in self.partitions[name]] for name in SPLIT_NAMES
            },
        }


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise SplitValidationError("ratios must contain three positive values summing to 1.0")


def create_split_manifest(
    records: Sequence[DatasetRecord],
    seed: int,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> SplitManifest:
    """Create a stable seeded split after sorting source records by identity."""
    if seed < 0:
        raise SplitValidationError("seed must be non-negative")
    _validate_ratios(ratios)
    entries = [
        SplitEntry(
            image_id=record.image_id,
            image_file=record.image_path.name,
            mask_file=record.mask_path.name,
            sha256=record.sha256,
        )
        for record in sorted(records, key=lambda item: item.image_id)
    ]
    random.Random(seed).shuffle(entries)
    total = len(entries)
    train_count = round(total * ratios[0])
    validation_count = int(total * ratios[1])
    partitions = {
        "train": tuple(entries[:train_count]),
        "validation": tuple(entries[train_count : train_count + validation_count]),
        "test": tuple(entries[train_count + validation_count :]),
    }
    manifest = SplitManifest(
        seed=seed,
        ratios=dict(zip(SPLIT_NAMES, ratios, strict=True)),
        partitions=partitions,
    )
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: SplitManifest) -> None:
    """Prove that all partitions are present and share no ID or content hash."""
    if set(manifest.partitions) != set(SPLIT_NAMES):
        raise SplitValidationError(f"manifest must contain exactly {SPLIT_NAMES}")
    if any(not manifest.partitions[name] for name in SPLIT_NAMES):
        raise SplitValidationError("every split must contain at least one image")
    id_owner: dict[str, str] = {}
    hash_owner: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for entry in manifest.partitions[split_name]:
            previous_id = id_owner.setdefault(entry.image_id, split_name)
            if previous_id != split_name:
                raise SplitValidationError(
                    f"image ID overlap: {entry.image_id} appears in {previous_id} and {split_name}"
                )
            previous_hash = hash_owner.setdefault(entry.sha256, split_name)
            if previous_hash != split_name:
                raise SplitValidationError(
                    "content hash overlap: "
                    f"{entry.sha256} appears in {previous_hash} and {split_name}"
                )


def save_split_manifest(manifest: SplitManifest, json_path: Path | str) -> None:
    """Write canonical JSON and a row-oriented CSV alongside it."""
    validate_split_manifest(manifest)
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["split", "image_id", "image_file", "mask_file", "sha256"]
        )
        writer.writeheader()
        for split_name in SPLIT_NAMES:
            for entry in manifest.partitions[split_name]:
                writer.writerow({"split": split_name, **asdict(entry)})


def load_split_manifest(path: Path | str) -> SplitManifest:
    """Load a JSON manifest and re-run leakage validation."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = SplitManifest(
            seed=int(payload["seed"]),
            ratios={name: float(payload["ratios"][name]) for name in SPLIT_NAMES},
            partitions={
                name: tuple(SplitEntry(**entry) for entry in payload["partitions"][name])
                for name in SPLIT_NAMES
            },
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SplitValidationError(f"malformed split manifest {manifest_path}: {exc}") from exc
    validate_split_manifest(manifest)
    return manifest
