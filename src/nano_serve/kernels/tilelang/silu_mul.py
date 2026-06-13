"""TileLang SiLU-mul entrypoint with torch fallback."""

from __future__ import annotations

from typing import Any

from nano_serve.kernels.tilelang.availability import check_tilelang_available
from nano_serve.kernels.torch_ops import silu_mul as torch_silu_mul


def silu_mul(gate: Any, up: Any, *, require_tilelang: bool = False) -> Any:
    availability = check_tilelang_available()
    if require_tilelang and not availability.available:
        raise RuntimeError(f"TileLang SiLU-mul is unavailable: {availability.error}")
    return torch_silu_mul(gate, up)

