"""Integrity checks for the Penn-Fudan image and mask layout."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


class DatasetValidationError(ValueError):
    """Raised when dataset structure or image content is incomplete."""


@dataclass(frozen=True)
class DatasetRecord:
    """A validated image/mask pair with a content identity."""

    image_id: str
    image_path: Path
    mask_path: Path
    sha256: str


@dataclass(frozen=True)
class DatasetSummary:
    """Validated Penn-Fudan inventory."""

    root: Path
    count: int
    records: tuple[DatasetRecord, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_root(root: Path | str, expected_count: int = 170) -> DatasetSummary:
    """Validate an extracted Penn-Fudan root and return paired records."""
    dataset_root = Path(root)
    image_dir = dataset_root / "PNGImages"
    mask_dir = dataset_root / "PedMasks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise DatasetValidationError(
            f"expected PNGImages and PedMasks directories under {dataset_root}"
        )

    images = {path.stem: path for path in sorted(image_dir.glob("*.png"))}
    masks = {
        path.stem.removesuffix("_mask"): path for path in sorted(mask_dir.glob("*_mask.png"))
    }
    if set(images) != set(masks):
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        raise DatasetValidationError(
            "image/mask pairing mismatch: "
            f"missing masks={missing_masks[:5]}, missing images={missing_images[:5]}"
        )
    if len(images) != expected_count:
        raise DatasetValidationError(
            f"expected {expected_count} image/mask pairs, found {len(images)} in {dataset_root}"
        )

    records: list[DatasetRecord] = []
    for image_id in sorted(images):
        image_path = images[image_id]
        mask_path = masks[image_id]
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(mask_path) as mask:
                mask.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise DatasetValidationError(f"unreadable image pair for {image_id}: {exc}") from exc
        records.append(
            DatasetRecord(
                image_id=image_id,
                image_path=image_path,
                mask_path=mask_path,
                sha256=_sha256(image_path),
            )
        )
    return DatasetSummary(root=dataset_root, count=len(records), records=tuple(records))
