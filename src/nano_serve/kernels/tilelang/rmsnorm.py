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
    if _can_use_tilelang_rmsnorm(x, weight, eps=eps, zero_centered=zero_centered):
        return _tilelang_rmsnorm(x, weight, eps=eps)
    if require_tilelang:
        raise RuntimeError("TileLang RMSNorm does not support this shape")
    return torch_rmsnorm(x, weight, eps=eps, zero_centered=zero_centered)


def _can_use_tilelang_rmsnorm(x: Any, weight: Any, *, eps: float, zero_centered: bool) -> bool:
    try:
        import torch
    except Exception:
        return False
    if zero_centered or eps <= 0.0:
        return False
    if not torch.is_tensor(x) or not torch.is_tensor(weight):
        return False
    if not check_tilelang_available().available:
        return False
    return (
        x.device.type == "cuda"
        and weight.device.type == "cuda"
        and x.dtype is torch.float16
        and weight.dtype is torch.float16
        and x.ndim >= 2
        and weight.ndim == 1
        and x.shape[-1] == weight.shape[0]
    )


def _tilelang_rmsnorm(x: Any, weight: Any, *, eps: float) -> Any:
    import torch

    from nano_serve.kernels.tilelang.simple_ops_kernel import cached_rmsnorm_kernel

    hidden_size = int(x.shape[-1])
    flat_x = x.contiguous().view(-1, hidden_size)
    output = torch.empty_like(flat_x)
    kernel = cached_rmsnorm_kernel(int(flat_x.shape[0]), hidden_size, float(eps))
    kernel(flat_x, weight.contiguous(), output)
    return output.view_as(x)

