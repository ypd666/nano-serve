"""Tracing helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
import warnings


def start_trace(request_id: str) -> str:
    return request_id


@contextmanager
def nvtx_range(name: str, *, enabled: bool = True) -> Iterator[None]:
    """Open an NVTX range when CUDA NVTX is available.

    CPU-only development and environments without torch keep this as a no-op so
    core imports stay portable.
    """

    if not enabled:
        yield
        return

    range_context = _torch_nvtx_range(name)
    if range_context is None:
        yield
        return

    with range_context:
        yield


def _torch_nvtx_range(name: str) -> Any | None:
    try:
        import torch
    except Exception:
        return None

    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if cuda is None or not callable(is_available):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            available = bool(is_available())
        if not available:
            return None
    except Exception:
        return None

    nvtx = getattr(cuda, "nvtx", None)
    range_factory = getattr(nvtx, "range", None)
    if callable(range_factory):
        try:
            return range_factory(name)
        except Exception:
            return None
    return None
