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
    if _can_use_tilelang_sample(logits, top_k=top_k, top_p=top_p):
        return _tilelang_sample(logits, top_k=top_k)
    if require_tilelang:
        raise RuntimeError("TileLang sampling helper does not support this shape")
    return top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)


def _can_use_tilelang_sample(logits: Any, *, top_k: int | None, top_p: float | None) -> bool:
    try:
        import torch
    except Exception:
        return False
    if not torch.is_tensor(logits):
        return False
    if not check_tilelang_available().available:
        return False
    return (
        logits.device.type == "cuda"
        and logits.dtype is torch.float16
        and logits.ndim == 1
        and logits.numel() > 0
        and top_k is not None
        and 0 < top_k <= logits.numel()
        and (top_p is None or top_p >= 1.0)
    )


def _tilelang_sample(logits: Any, *, top_k: int | None) -> Any:
    import torch

    from nano_serve.kernels.tilelang.simple_ops_kernel import cached_topk_filter_kernel

    if top_k is None:
        raise ValueError("top_k is required for TileLang sampling helper")
    flat_logits = logits.contiguous()
    output = torch.empty_like(flat_logits)
    kernel = cached_topk_filter_kernel(int(flat_logits.numel()), int(top_k))
    kernel(flat_logits, output)
    output = output.masked_fill(output == torch.finfo(output.dtype).min, float("-inf"))
    return output

