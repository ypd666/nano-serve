"""TileLang paged attention entrypoint with torch fallback."""

from __future__ import annotations

from typing import Any

from nano_serve.attention.paged_gather_torch import TorchGatherPagedAttention
from nano_serve.kernels.tilelang.availability import check_tilelang_available


def paged_decode_attention(
    query: Any,
    paged_key: Any,
    paged_value: Any,
    block_tables: Any,
    seq_lens: Any,
    *,
    require_tilelang: bool = False,
    scale: float | None = None,
) -> tuple[Any, Any]:
    availability = check_tilelang_available()
    if require_tilelang and not availability.available:
        raise RuntimeError(f"TileLang paged attention is unavailable: {availability.error}")
    return TorchGatherPagedAttention().forward_decode(
        query,
        paged_key,
        paged_value,
        block_tables,
        seq_lens,
        scale=scale,
    )

