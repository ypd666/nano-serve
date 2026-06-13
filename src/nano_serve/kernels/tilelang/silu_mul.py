"""TileLang SiLU-mul entrypoint with torch fallback."""

from __future__ import annotations

from typing import Any

from nano_serve.kernels.tilelang.availability import check_tilelang_available
from nano_serve.kernels.torch_ops import silu_mul as torch_silu_mul


def silu_mul(gate: Any, up: Any, *, require_tilelang: bool = False) -> Any:
    availability = check_tilelang_available()
    if require_tilelang and not availability.available:
        raise RuntimeError(f"TileLang SiLU-mul is unavailable: {availability.error}")
    if _can_use_tilelang_silu_mul(gate, up):
        return _tilelang_silu_mul(gate, up)
    if require_tilelang:
        raise RuntimeError("TileLang SiLU-mul does not support this shape")
    return torch_silu_mul(gate, up)


def _can_use_tilelang_silu_mul(gate: Any, up: Any) -> bool:
    try:
        import torch
    except Exception:
        return False
    if not torch.is_tensor(gate) or not torch.is_tensor(up):
        return False
    if not check_tilelang_available().available:
        return False
    return (
        gate.device.type == "cuda"
        and up.device.type == "cuda"
        and gate.dtype is torch.float16
        and up.dtype is torch.float16
        and gate.shape == up.shape
        and gate.numel() > 0
    )


def _tilelang_silu_mul(gate: Any, up: Any) -> Any:
    import torch

    from nano_serve.kernels.tilelang.simple_ops_kernel import cached_silu_mul_kernel

    flat_gate = gate.contiguous().view(-1)
    flat_up = up.contiguous().view(-1)
    output = torch.empty_like(flat_gate)
    kernel = cached_silu_mul_kernel(int(flat_gate.numel()))
    kernel(flat_gate, flat_up, output)
    return output.view_as(gate)

