"""TileLang paged attention entrypoint with torch fallback."""

from __future__ import annotations

import time
from typing import Any

from nano_serve.attention.paged_gather_torch import (
    PagedAttentionMetadata,
    TorchGatherPagedAttention,
)
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
    if can_use_tilelang_decode(query, paged_key, paged_value, block_tables, seq_lens, scale):
        return _tilelang_forward_decode(query, paged_key, paged_value, block_tables, seq_lens)
    if require_tilelang:
        raise RuntimeError("TileLang paged attention does not support this decode shape")
    return TorchGatherPagedAttention().forward_decode(
        query,
        paged_key,
        paged_value,
        block_tables,
        seq_lens,
        scale=scale,
    )


def can_use_tilelang_decode(
    query: Any,
    paged_key: Any,
    paged_value: Any,
    block_tables: Any,
    seq_lens: Any,
    scale: float | None,
) -> bool:
    try:
        import torch
    except Exception:
        return False
    if scale is not None:
        return False
    if not torch.is_tensor(query) or not torch.is_tensor(paged_key) or not torch.is_tensor(paged_value):
        return False
    if not check_tilelang_available().available:
        return False
    if query.device.type != "cuda" or paged_key.device.type != "cuda" or paged_value.device.type != "cuda":
        return False
    if query.dtype is not torch.float16 or paged_key.dtype is not torch.float16 or paged_value.dtype is not torch.float16:
        return False
    if query.ndim != 4 or paged_key.ndim != 4 or paged_value.ndim != 4:
        return False
    if query.shape[2] != 1 or paged_key.shape != paged_value.shape:
        return False
    if query.shape[3] != paged_key.shape[3]:
        return False
    if query.shape[1] % paged_key.shape[1] != 0:
        return False
    if len(block_tables) != query.shape[0] or len(seq_lens) != query.shape[0]:
        return False
    return all(int(seq_len) > 0 for seq_len in seq_lens)


def _tilelang_forward_decode(
    query: Any,
    paged_key: Any,
    paged_value: Any,
    block_tables: Any,
    seq_lens: Any,
) -> tuple[Any, PagedAttentionMetadata]:
    import torch

    from nano_serve.kernels.tilelang.paged_decode_kernel import cached_paged_decode_kernel

    if paged_key.shape[2] <= 0:
        raise ValueError("block size must be positive")
    batch_size = int(query.shape[0])
    query_heads = int(query.shape[1])
    kv_heads = int(paged_key.shape[1])
    head_dim = int(query.shape[3])
    block_size = int(paged_key.shape[2])
    context_len = max(int(seq_len) for seq_len in seq_lens)
    max_blocks = max(len(block_ids) for block_ids in block_tables)
    num_blocks = int(paged_key.shape[0])
    block_table_tensor = torch.zeros(
        (batch_size, max_blocks),
        device=query.device,
        dtype=torch.int32,
    )
    for batch_index, block_ids in enumerate(block_tables):
        if len(block_ids) < (int(seq_lens[batch_index]) + block_size - 1) // block_size:
            raise ValueError("block table is too short for sequence length")
        if block_ids:
            block_table_tensor[batch_index, : len(block_ids)] = torch.as_tensor(
                block_ids,
                device=query.device,
                dtype=torch.int32,
            )
    seq_lens_tensor = torch.as_tensor(seq_lens, device=query.device, dtype=torch.int32)
    output = torch.empty_like(query)
    kernel = cached_paged_decode_kernel(
        batch_size,
        query_heads,
        kv_heads,
        head_dim,
        context_len,
        block_size,
        max_blocks,
        num_blocks,
    )
    if query.device.type == "cuda":
        torch.cuda.synchronize()
    start_ns = time.monotonic_ns()
    kernel(query, paged_key, paged_value, block_table_tensor, seq_lens_tensor, output)
    if query.device.type == "cuda":
        torch.cuda.synchronize()
    attention_ms = (time.monotonic_ns() - start_ns) / 1_000_000
    return output, PagedAttentionMetadata(
        gather_time_ms=0.0,
        attention_time_ms=attention_ms,
        context_tokens=context_len,
        block_size=block_size,
        batch_size=batch_size,
    )

