"""Single-request contiguous KV cache for the Phase 2 torch reference path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nano_serve.kv_cache.base import KVHandle


@dataclass(frozen=True)
class ContiguousKVCacheConfig:
    max_model_len: int
    num_layers: int
    block_size: int = 16


@dataclass
class ContiguousLayerState:
    layer_type: str
    key: Any | None = None
    value: Any | None = None
    conv_state: Any | None = None
    recurrent_state: Any | None = None


@dataclass
class ContiguousRequestState:
    handle: KVHandle
    max_model_len: int
    layer_states: list[ContiguousLayerState]
    sequence_length: int = 0


@dataclass(frozen=True)
class ContiguousKVCacheStats:
    requests: int
    tokens_used: int
    tokens_capacity: int
    bytes_used: int
    bytes_capacity: int
    fragmentation: float

    def to_dict(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "tokens_used": self.tokens_used,
            "tokens_capacity": self.tokens_capacity,
            "bytes_used": self.bytes_used,
            "bytes_capacity": self.bytes_capacity,
            "fragmentation": self.fragmentation,
        }


class ContiguousKVCache:
    def __init__(
        self,
        config: ContiguousKVCacheConfig | None = None,
        *,
        num_layers: int | None = None,
        max_tokens: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        dtype: Any | None = None,
        device: Any | None = None,
        block_size: int = 16,
    ) -> None:
        if config is None:
            if max_tokens is None or num_layers is None:
                raise TypeError("config or max_tokens/num_layers is required")
            config = ContiguousKVCacheConfig(
                max_model_len=max_tokens,
                num_layers=num_layers,
                block_size=block_size,
            )
        self.config = config
        self._legacy_num_heads = num_heads
        self._legacy_head_dim = head_dim
        self._legacy_dtype = dtype
        self._legacy_device = device
        self._reserved_tokens = 0
        self._request_capacity: dict[str, int] = {}
        self._legacy_kv: dict[str, tuple[Any, Any]] = {}
        self._requests: dict[str, ContiguousRequestState] = {}

    def allocate_prefill(
        self,
        request_id: str,
        n_tokens: int,
        *,
        max_decode_tokens: int = 0,
        layer_states: list[ContiguousLayerState] | None = None,
    ) -> KVHandle:
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        capacity = n_tokens + max_decode_tokens
        if capacity > self.config.max_model_len:
            raise ValueError(
                f"prefill tokens exceed cache capacity: {capacity} > {self.config.max_model_len}"
            )
        if request_id not in self._requests:
            if self._reserved_tokens + capacity > self.config.max_model_len:
                raise MemoryError("out of contiguous KV capacity")
            self._reserved_tokens += capacity
            self._request_capacity[request_id] = capacity
        states = layer_states or [
            ContiguousLayerState(layer_type="unknown") for _ in range(self.config.num_layers)
        ]
        handle = KVHandle(
            request_id=request_id,
            num_tokens=n_tokens,
            block_ids=list(range(_ceil_div(n_tokens, self.config.block_size))),
        )
        self._requests[request_id] = ContiguousRequestState(
            handle=handle,
            max_model_len=self.config.max_model_len,
            layer_states=states,
            sequence_length=n_tokens,
        )
        return handle

    def allocate_decode_slot(self, request_id: str) -> KVHandle:
        state = self._request(request_id)
        next_length = state.sequence_length + 1
        capacity = self._request_capacity.get(request_id, state.max_model_len)
        if next_length > capacity:
            raise ValueError(
                f"decode token exceeds cache capacity: {next_length} > {capacity}"
            )
        state.sequence_length = next_length
        state.handle.num_tokens = next_length
        state.handle.block_ids = list(range(_ceil_div(next_length, self.config.block_size)))
        return state.handle

    def set_layer_states(
        self,
        request_id: str,
        layer_states: list[ContiguousLayerState],
        *,
        sequence_length: int,
    ) -> None:
        if len(layer_states) != self.config.num_layers:
            raise ValueError(
                f"expected {self.config.num_layers} layer states, got {len(layer_states)}"
            )
        state = self._request(request_id)
        state.layer_states = layer_states
        state.sequence_length = sequence_length
        state.handle.num_tokens = sequence_length
        state.handle.block_ids = list(range(_ceil_div(sequence_length, self.config.block_size)))

    def layer_states(self, request_id: str) -> list[ContiguousLayerState]:
        return self._request(request_id).layer_states

    def sequence_length(self, request_id: str) -> int:
        return self._request(request_id).sequence_length

    def free(self, request_id: str) -> None:
        state = self._requests.pop(request_id, None)
        self._legacy_kv.pop(request_id, None)
        if state is not None:
            self._reserved_tokens -= self._request_capacity.pop(request_id, 0)

    def get_block_table(self, request_id: str) -> list[int]:
        state = self._requests.get(request_id)
        if state is None:
            return []
        return list(state.handle.block_ids)

    def stats(self) -> ContiguousKVCacheStats:
        requests = list(self._requests.values())
        tokens_used = sum(request.sequence_length for request in requests)
        tokens_capacity = sum(
            self._request_capacity.get(request.handle.request_id, request.max_model_len)
            for request in requests
        )
        bytes_used = sum(_layer_state_bytes(layer) for request in requests for layer in request.layer_states)
        bytes_capacity = bytes_used
        fragmentation = 0.0
        if tokens_capacity:
            fragmentation = 1.0 - (tokens_used / tokens_capacity)
        return ContiguousKVCacheStats(
            requests=len(requests),
            tokens_used=tokens_used,
            tokens_capacity=tokens_capacity,
            bytes_used=bytes_used,
            bytes_capacity=bytes_capacity,
            fragmentation=fragmentation,
        )

    def _request(self, request_id: str) -> ContiguousRequestState:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown KV request: {request_id}") from exc

    def write_prefill(self, request_id: str, keys: Any, values: Any) -> None:
        state = self._request(request_id)
        expected_shape = self._legacy_shape(state.sequence_length)
        if tuple(keys.shape) != expected_shape or tuple(values.shape) != expected_shape:
            raise ValueError(
                f"prefill KV shape must be {expected_shape}, got {tuple(keys.shape)} and {tuple(values.shape)}"
            )
        self._legacy_kv[request_id] = (keys.detach().clone(), values.detach().clone())

    def append_decode(self, request_id: str, keys: Any, values: Any) -> KVHandle:
        self._request(request_id)
        expected_shape = self._legacy_shape(1)
        if tuple(keys.shape) != expected_shape or tuple(values.shape) != expected_shape:
            raise ValueError(
                f"decode KV shape must be {expected_shape}, got {tuple(keys.shape)} and {tuple(values.shape)}"
            )
        current_keys, current_values = self.get_kv(request_id)
        handle = self.allocate_decode_slot(request_id)
        import torch

        self._legacy_kv[request_id] = (
            torch.cat((current_keys, keys.detach().clone()), dim=1),
            torch.cat((current_values, values.detach().clone()), dim=1),
        )
        return handle

    def get_kv(self, request_id: str) -> tuple[Any, Any]:
        self._request(request_id)
        try:
            return self._legacy_kv[request_id]
        except KeyError as exc:
            raise KeyError(request_id) from exc

    def _legacy_shape(self, num_tokens: int) -> tuple[int, int, int, int]:
        if self._legacy_num_heads is None or self._legacy_head_dim is None:
            raise ValueError("legacy tensor KV API requires num_heads and head_dim")
        return (
            self.config.num_layers,
            num_tokens,
            self._legacy_num_heads,
            self._legacy_head_dim,
        )


def _layer_state_bytes(state: ContiguousLayerState) -> int:
    total = 0
    for tensor in (state.key, state.value, state.conv_state, state.recurrent_state):
        if tensor is not None:
            total += int(tensor.numel() * tensor.element_size())
    return total


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
