"""TileLang RoPE entrypoint with torch fallback."""

from __future__ import annotations

from typing import Any

from nano_serve.kernels.tilelang.availability import check_tilelang_available
from nano_serve.kernels.torch_ops import rope as torch_rope


def rope(
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    *,
    require_tilelang: bool = False,
) -> tuple[Any, Any]:
    availability = check_tilelang_available()
    if require_tilelang and not availability.available:
        raise RuntimeError(f"TileLang RoPE is unavailable: {availability.error}")
    return torch_rope(q, k, cos, sin)

