"""TileLang sampling helper entrypoint with torch fallback."""

from __future__ import annotations

from typing import Any

from nano_serve.kernels.tilelang.availability import check_tilelang_available
from nano_serve.kernels.torch_ops import top_k_top_p_filter


def sample(
    logits: Any,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    require_tilelang: bool = False,
) -> Any:
    availability = check_tilelang_available()
    if require_tilelang and not availability.available:
        raise RuntimeError(f"TileLang sampling helper is unavailable: {availability.error}")
    return top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

