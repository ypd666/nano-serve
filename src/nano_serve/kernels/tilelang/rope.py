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
    if _can_use_tilelang_rope(q, k, cos, sin):
        return _tilelang_rope(q, k, cos, sin)
    if require_tilelang:
        raise RuntimeError("TileLang RoPE does not support this shape")
    return torch_rope(q, k, cos, sin)


def _can_use_tilelang_rope(q: Any, k: Any, cos: Any, sin: Any) -> bool:
    try:
        import torch
    except Exception:
        return False
    if not all(torch.is_tensor(tensor) for tensor in (q, k, cos, sin)):
        return False
    if not check_tilelang_available().available:
        return False
    return (
        q.device.type == "cuda"
        and k.device.type == "cuda"
        and cos.device.type == "cuda"
        and sin.device.type == "cuda"
        and q.dtype is torch.float16
        and k.dtype is torch.float16
        and cos.dtype is torch.float16
        and sin.dtype is torch.float16
        and q.ndim == 4
        and k.ndim == 4
        and cos.ndim == 3
        and sin.ndim == 3
        and q.shape[0] == k.shape[0]
        and cos.shape[0] == sin.shape[0]
        and cos.shape[0] in {1, q.shape[0]}
        and q.shape[2] == k.shape[2] == cos.shape[1] == sin.shape[1]
        and q.shape[3] == k.shape[3] == cos.shape[2] == sin.shape[2]
        and q.shape[3] % 2 == 0
    )


def _tilelang_rope(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    import torch

    from nano_serve.kernels.tilelang.simple_ops_kernel import (
        cached_rope_key_kernel,
        cached_rope_query_kernel,
    )

    batch_size, query_heads, seq_len, head_dim = q.shape
    kv_heads = int(k.shape[1])
    total_tokens = int(batch_size * seq_len)
    if cos.shape[0] == 1 and batch_size != 1:
        cos = cos.expand(batch_size, -1, -1)
        sin = sin.expand(batch_size, -1, -1)
    q_flat = q.permute(0, 2, 1, 3).contiguous().view(total_tokens, query_heads, head_dim)
    k_flat = k.permute(0, 2, 1, 3).contiguous().view(total_tokens, kv_heads, head_dim)
    cos_flat = cos.contiguous().view(total_tokens, head_dim)
    sin_flat = sin.contiguous().view(total_tokens, head_dim)
    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)
    query_kernel = cached_rope_query_kernel(total_tokens, int(query_heads), int(head_dim))
    key_kernel = cached_rope_key_kernel(total_tokens, kv_heads, int(head_dim))
    query_kernel(q_flat, cos_flat, sin_flat, q_out)
    key_kernel(k_flat, cos_flat, sin_flat, k_out)
    return (
        q_out.view(batch_size, seq_len, query_heads, head_dim).permute(0, 2, 1, 3).contiguous(),
        k_out.view(batch_size, seq_len, kv_heads, head_dim).permute(0, 2, 1, 3).contiguous(),
    )

