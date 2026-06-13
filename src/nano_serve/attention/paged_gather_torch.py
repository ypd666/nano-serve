"""Torch gather-based paged attention reference."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PagedAttentionMetadata:
    gather_time_ms: float
    attention_time_ms: float
    context_tokens: int
    block_size: int
    batch_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "gather_time_ms": self.gather_time_ms,
            "attention_time_ms": self.attention_time_ms,
            "context_tokens": self.context_tokens,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
        }


class TorchGatherPagedAttention:
    def gather_kv(
        self,
        paged_key,
        paged_value,
        block_tables,
        seq_lens,
    ):
        """Gather paged K/V blocks into padded contiguous tensors.

        Args:
            paged_key/value: `(num_blocks, num_kv_heads, block_size, head_dim)`.
            block_tables: list of block id lists, one per request.
            seq_lens: sequence lengths for each request.

        Returns:
            `(key, value)` tensors shaped
            `(batch, num_kv_heads, max_seq_len, head_dim)`.
        """
        import torch

        if len(block_tables) != len(seq_lens):
            raise ValueError("block_tables and seq_lens must have the same length")
        if paged_key.shape != paged_value.shape:
            raise ValueError("paged key/value tensors must have the same shape")
        if paged_key.ndim != 4:
            raise ValueError("paged key/value tensors must be 4D")

        batch_size = len(block_tables)
        _, num_kv_heads, block_size, head_dim = paged_key.shape
        max_seq_len = max(seq_lens, default=0)
        key = torch.zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            device=paged_key.device,
            dtype=paged_key.dtype,
        )
        value = torch.zeros_like(key)
        if max_seq_len == 0:
            return key, value

        for batch_index, (block_ids, seq_len) in enumerate(zip(block_tables, seq_lens, strict=True)):
            if seq_len < 0:
                raise ValueError("sequence lengths must be non-negative")
            expected_blocks = (seq_len + block_size - 1) // block_size
            if len(block_ids) < expected_blocks:
                raise ValueError(
                    f"block table too short for seq_len={seq_len}: "
                    f"{len(block_ids)} < {expected_blocks}"
                )
            write_offset = 0
            for block_id in block_ids[:expected_blocks]:
                block_tokens = min(block_size, seq_len - write_offset)
                if block_tokens <= 0:
                    break
                key[
                    batch_index,
                    :,
                    write_offset : write_offset + block_tokens,
                    :,
                ] = paged_key[block_id, :, :block_tokens, :]
                value[
                    batch_index,
                    :,
                    write_offset : write_offset + block_tokens,
                    :,
                ] = paged_value[block_id, :, :block_tokens, :]
                write_offset += block_tokens
        return key, value

    def forward_prefill(
        self,
        query,
        key,
        value,
        *,
        scale: float | None = None,
        causal: bool = True,
    ):
        """Contiguous prefill attention reference.

        Shapes:
            query: `(batch, query_heads, seq_len, head_dim)`
            key/value: `(batch, kv_heads, seq_len, head_dim)`
        """
        start_ns = time.monotonic_ns()
        output = _attention(query, key, value, scale=scale, causal=causal)
        attention_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        return output, PagedAttentionMetadata(
            gather_time_ms=0.0,
            attention_time_ms=attention_ms,
            context_tokens=int(query.shape[-2]),
            block_size=int(key.shape[-2]),
            batch_size=int(query.shape[0]),
        )

    def forward_decode(
        self,
        query,
        paged_key,
        paged_value,
        block_tables,
        seq_lens,
        *,
        scale: float | None = None,
    ):
        """Decode one or more query tokens against paged K/V context."""
        gather_start_ns = time.monotonic_ns()
        key, value = self.gather_kv(paged_key, paged_value, block_tables, seq_lens)
        gather_ms = (time.monotonic_ns() - gather_start_ns) / 1_000_000
        attention_start_ns = time.monotonic_ns()
        output = _attention(query, key, value, scale=scale, causal=False, key_lens=seq_lens)
        attention_ms = (time.monotonic_ns() - attention_start_ns) / 1_000_000
        return output, PagedAttentionMetadata(
            gather_time_ms=gather_ms,
            attention_time_ms=attention_ms,
            context_tokens=max(seq_lens, default=0),
            block_size=int(paged_key.shape[2]),
            batch_size=len(block_tables),
        )


def _attention(query, key, value, *, scale: float | None, causal: bool, key_lens=None):
    import torch

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be 4D")
    if key.shape != value.shape:
        raise ValueError("key/value shapes must match")
    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key batch sizes must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")

    key = _repeat_kv(key, query.shape[1])
    value = _repeat_kv(value, query.shape[1])
    scale = scale if scale is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if causal:
        query_len = query.shape[-2]
        key_len = key.shape[-2]
        mask = torch.ones((query_len, key_len), dtype=torch.bool, device=query.device)
        mask = torch.triu(mask, diagonal=1 + key_len - query_len)
        scores = scores.masked_fill(mask.view(1, 1, query_len, key_len), torch.finfo(scores.dtype).min)
    if key_lens is not None:
        if len(key_lens) != query.shape[0]:
            raise ValueError("key_lens length must match batch size")
        positions = torch.arange(key.shape[-2], device=query.device).view(1, 1, 1, -1)
        lengths = torch.as_tensor(key_lens, device=query.device).view(-1, 1, 1, 1)
        scores = scores.masked_fill(positions >= lengths, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value)


def _repeat_kv(hidden_states, target_heads: int):
    if hidden_states.shape[1] == target_heads:
        return hidden_states
    if target_heads % hidden_states.shape[1] != 0:
        raise ValueError(
            f"target_heads must be a multiple of kv heads: {target_heads} vs {hidden_states.shape[1]}"
        )
    repeats = target_heads // hidden_states.shape[1]
    return hidden_states.repeat_interleave(repeats, dim=1)

