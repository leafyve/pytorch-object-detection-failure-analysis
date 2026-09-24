"""Runtime device selection and environment provenance."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any

import torch


def select_device() -> torch.device:
    """Choose CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect non-sensitive software and hardware metadata for a run."""
    device = select_device()
    device_metadata: dict[str, Any] = {"type": device.type}
    if device.type == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_metadata.update(
            {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torchvision_version": _version("torchvision"),
        "numpy_version": _version("numpy"),
        "device": device_metadata,
    }
