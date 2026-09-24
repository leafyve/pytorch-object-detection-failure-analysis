from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from podfa.data.download import DownloadError, download_dataset
from podfa.data.validation import DatasetValidationError, validate_dataset_root


def _write_png(path: Path, *, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L" if mask else "RGB", (8, 6), 1 if mask else (10, 20, 30))
    image.save(path)


def _create_dataset_tree(root: Path, count: int = 2) -> Path:
    dataset_root = root / "PennFudanPed"
    for index in range(count):
        image_id = f"FudanPed{index:05d}"
        _write_png(dataset_root / "PNGImages" / f"{image_id}.png")
        _write_png(dataset_root / "PedMasks" / f"{image_id}_mask.png", mask=True)
    return dataset_root


def _create_archive(tmp_path: Path, count: int = 2) -> Path:
    source = tmp_path / "source"
    dataset_root = _create_dataset_tree(source, count)
    archive = tmp_path / "PennFudanPed.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for file_path in dataset_root.rglob("*.png"):
            bundle.write(file_path, file_path.relative_to(source))
    return archive


def test_validate_dataset_root_returns_paired_records(tmp_path: Path) -> None:
    root = _create_dataset_tree(tmp_path, count=2)

    summary = validate_dataset_root(root, expected_count=2)

    assert summary.count == 2
    assert [record.image_id for record in summary.records] == ["FudanPed00000", "FudanPed00001"]
    assert all(len(record.sha256) == 64 for record in summary.records)


def test_validate_dataset_root_rejects_missing_mask(tmp_path: Path) -> None:
    root = _create_dataset_tree(tmp_path, count=2)
    (root / "PedMasks" / "FudanPed00001_mask.png").unlink()

    with pytest.raises(DatasetValidationError, match="pairing"):
        validate_dataset_root(root, expected_count=2)


def test_download_extracts_valid_archive_and_records_source(tmp_path: Path) -> None:
    archive = _create_archive(tmp_path, count=2)
    destination = tmp_path / "downloaded"

    summary = download_dataset(archive.as_uri(), destination, expected_count=2)

    assert summary.count == 2
    assert (destination / "PennFudanPed" / "source_metadata.json").is_file()


def test_download_refuses_to_overwrite_incomplete_destination(tmp_path: Path) -> None:
    archive = _create_archive(tmp_path, count=2)
    destination = tmp_path / "downloaded"
    (destination / "PennFudanPed" / "PNGImages").mkdir(parents=True)

    with pytest.raises(DownloadError, match="refusing to overwrite"):
        download_dataset(archive.as_uri(), destination, expected_count=2)


def test_download_rejects_corrupt_archive_without_partial_dataset(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip")
    destination = tmp_path / "downloaded"

    with pytest.raises(DownloadError, match="valid ZIP"):
        download_dataset(archive.as_uri(), destination, expected_count=2)

    assert not (destination / "PennFudanPed").exists()


def test_download_supplies_trusted_ssl_context_for_https(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _create_archive(tmp_path, count=1)
    payload = archive.read_bytes()

    class Response:
        def __enter__(self) -> Response:
            self._stream = __import__("io").BytesIO(payload)
            return self

        def __exit__(self, *_args: Any) -> None:
            self._stream.close()

        def read(self, size: int = -1) -> bytes:
            return self._stream.read(size)

    observed: dict[str, Any] = {}

    def fake_urlopen(url: str, *, timeout: int, context: Any) -> Response:
        observed.update(url=url, timeout=timeout, context=context)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    summary = download_dataset(
        "https://example.test/PennFudanPed.zip", tmp_path / "downloaded", expected_count=1
    )

    assert summary.count == 1
    assert observed["url"] == "https://example.test/PennFudanPed.zip"
    assert observed["context"].verify_mode == __import__("ssl").CERT_REQUIRED
