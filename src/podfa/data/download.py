"""Atomic downloader for the public Penn-Fudan pedestrian dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import certifi

from podfa.data.validation import DatasetSummary, DatasetValidationError, validate_dataset_root

PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


class DownloadError(RuntimeError):
    """Raised when acquisition cannot produce a validated atomic dataset."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise DownloadError(f"archive contains unsafe path: {member.filename}")
    archive.extractall(destination)


def _open_url(url: str, timeout: int):  # type: ignore[no-untyped-def]
    if urlparse(url).scheme.lower() == "https":
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(url, timeout=timeout, context=context)
    return urllib.request.urlopen(url, timeout=timeout)


def download_dataset(
    url: str,
    destination: Path | str,
    expected_sha256: str | None = None,
    *,
    expected_count: int = 170,
) -> DatasetSummary:
    """Download, verify, and atomically install Penn-Fudan under destination."""
    destination_path = Path(destination)
    final_root = destination_path / "PennFudanPed"
    if final_root.exists():
        try:
            return validate_dataset_root(final_root, expected_count=expected_count)
        except DatasetValidationError as exc:
            raise DownloadError(
                f"refusing to overwrite incomplete or corrupted dataset at {final_root}: {exc}"
            ) from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = destination_path.parent
    with tempfile.TemporaryDirectory(prefix="penn-fudan-", dir=staging_parent) as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / "PennFudanPed.zip"
        try:
            with (
                _open_url(url, timeout=120) as response,
                archive_path.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
        except (OSError, urllib.error.URLError) as exc:
            raise DownloadError(f"failed to download {url}: {exc}") from exc

        archive_hash = _file_sha256(archive_path)
        if expected_sha256 and archive_hash.lower() != expected_sha256.lower():
            raise DownloadError(
                f"archive SHA-256 mismatch: expected {expected_sha256}, received {archive_hash}"
            )
        try:
            with zipfile.ZipFile(archive_path) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise DownloadError(f"ZIP integrity check failed at {bad_member}")
                extraction_root = temporary_path / "extracted"
                extraction_root.mkdir()
                _safe_extract(archive, extraction_root)
        except zipfile.BadZipFile as exc:
            raise DownloadError(f"download is not a valid ZIP archive: {exc}") from exc

        staged_root = extraction_root / "PennFudanPed"
        try:
            summary = validate_dataset_root(staged_root, expected_count=expected_count)
        except DatasetValidationError as exc:
            raise DownloadError(f"extracted dataset is incomplete: {exc}") from exc
        metadata = {
            "dataset": "Penn-Fudan Pedestrian Detection",
            "source_url": url,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "archive_sha256": archive_hash,
            "image_count": summary.count,
            "expected_layout": ["PennFudanPed/PNGImages", "PennFudanPed/PedMasks"],
            "attribution": "University of Pennsylvania Fudan University Pedestrian Database",
        }
        (staged_root / "source_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        destination_path.mkdir(parents=True, exist_ok=True)
        if final_root.exists():
            raise DownloadError(f"destination appeared during download: {final_root}")
        shutil.move(str(staged_root), str(final_root))

    return validate_dataset_root(final_root, expected_count=expected_count)
