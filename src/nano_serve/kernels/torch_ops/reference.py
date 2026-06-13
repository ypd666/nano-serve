"""Torch reference operators used by kernel benchmarks."""

from __future__ import annotations

from typing import Any


def rmsnorm(x: Any, weight: Any, *, eps: float = 1e-6, zero_centered: bool = False) -> Any:
    import torch

    tensor = torch.as_tensor(x)
    scale = torch.as_tensor(weight, device=tensor.device, dtype=tensor.dtype)
    if zero_centered:
        scale = scale + 1.0
    output = tensor.float()
    output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (output * scale.float()).to(tensor.dtype)


def rope(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    import torch

    query = torch.as_tensor(q)
    key = torch.as_tensor(k)
    cos_tensor = torch.as_tensor(cos, device=query.device, dtype=query.dtype)
    sin_tensor = torch.as_tensor(sin, device=query.device, dtype=query.dtype)
    while cos_tensor.ndim < query.ndim:
        cos_tensor = cos_tensor.unsqueeze(1)
        sin_tensor = sin_tensor.unsqueeze(1)
    rotary_dim = cos_tensor.shape[-1]
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    q_embed = (q_rot * cos_tensor) + (_rotate_half(q_rot) * sin_tensor)
    k_embed = (k_rot * cos_tensor) + (_rotate_half(k_rot) * sin_tensor)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


def silu_mul(gate: Any, up: Any) -> Any:
    import torch

    gate_tensor = torch.as_tensor(gate)
    up_tensor = torch.as_tensor(up, device=gate_tensor.device, dtype=gate_tensor.dtype)
    return torch.nn.functional.silu(gate_tensor.float()).to(gate_tensor.dtype) * up_tensor


def top_k_top_p_filter(logits: Any, *, top_k: int | None = None, top_p: float | None = None) -> Any:
    import torch

    logits_tensor = torch.as_tensor(logits)
    filtered = logits_tensor.clone().float()
    if filtered.ndim != 1:
        raise ValueError(f"logits must be 1D, got shape {tuple(filtered.shape)}")
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k < filtered.numel():
            values, _ = filtered.topk(top_k)
            filtered = filtered.masked_fill(filtered < values[-1], float("-inf"))
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_p < 1:
            sorted_logits, sorted_indices = filtered.sort(descending=True)
            sorted_probs = sorted_logits.softmax(dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            filtered_sorted = sorted_logits.masked_fill(remove_mask, float("-inf"))
            filtered = filtered.new_full(filtered.shape, float("-inf"))
            filtered = filtered.scatter(0, sorted_indices, filtered_sorted)
    return filtered.to(logits_tensor.dtype)


def _rotate_half(x: Any) -> Any:
    import torch

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
