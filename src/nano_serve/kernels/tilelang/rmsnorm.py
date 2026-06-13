"""TileLang RMSNorm entrypoint with torch fallback."""

from __future__ import annotations

from typing import Any

from nano_serve.kernels.tilelang.availability import check_tilelang_available
from nano_serve.kernels.torch_ops import rmsnorm as torch_rmsnorm


def rmsnorm(
    x: Any,
    weight: Any,
    *,
    eps: float = 1e-6,
    zero_centered: bool = False,
    require_tilelang: bool = False,
) -> Any:
    availability = check_tilelang_available()
    if require_tilelang and not availability.available:
        raise RuntimeError(f"TileLang RMSNorm is unavailable: {availability.error}")
    return torch_rmsnorm(x, weight, eps=eps, zero_centered=zero_centered)

