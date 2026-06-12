"""Runtime platform detection.

Shared infrastructure must stay usable on macOS Apple Silicon without CUDA.
Performance paths can use CUDA on Linux NVIDIA machines when available.
"""

from __future__ import annotations

import platform as platform_lib
import sys
from dataclasses import asdict, dataclass
from types import ModuleType


@dataclass(frozen=True)
class PlatformInfo:
    os_name: str
    system: str
    machine: str
    python_version: str
    torch_version: str | None
    device_backend: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: tuple[str, ...]
    is_macos: bool
    is_apple_silicon: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_log_fields(self) -> dict[str, object]:
        return self.to_dict()


def detect_platform(torch_module: object | None = None) -> PlatformInfo:
    torch = _import_torch() if torch_module is None else torch_module
    torch_version = _get_torch_version(torch)
    cuda_available = _cuda_available(torch)
    cuda_device_count = _cuda_device_count(torch) if cuda_available else 0
    cuda_device_names = (
        tuple(_cuda_device_name(torch, index) for index in range(cuda_device_count))
        if cuda_available
        else ()
    )
    system = platform_lib.system()
    machine = platform_lib.machine()

    return PlatformInfo(
        os_name=platform_lib.platform(),
        system=system,
        machine=machine,
        python_version=sys.version.split()[0],
        torch_version=torch_version,
        device_backend="cuda" if cuda_available else "cpu",
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_device_names,
        is_macos=system == "Darwin",
        is_apple_silicon=system == "Darwin" and machine in {"arm64", "aarch64"},
    )


def _import_torch() -> ModuleType | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _get_torch_version(torch: object | None) -> str | None:
    if torch is None:
        return None
    version = getattr(torch, "__version__", None)
    return str(version) if version is not None else None


def _cuda_available(torch: object | None) -> bool:
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


def _cuda_device_count(torch: object | None) -> int:
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    device_count = getattr(cuda, "device_count", None)
    if not callable(device_count):
        return 0
    return int(device_count())


def _cuda_device_name(torch: object | None, index: int) -> str:
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    get_device_name = getattr(cuda, "get_device_name", None)
    if not callable(get_device_name):
        return f"cuda:{index}"
    return str(get_device_name(index))

