"""Tensor-parallel reference plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorParallelCaseResult:
    mode: str
    world_size: int
    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    max_abs_diff: float
    all_reduce_bytes: int
    shard_parameter_bytes: int
    shard_shapes: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "world_size": self.world_size,
            "input_shape": list(self.input_shape),
            "weight_shape": list(self.weight_shape),
            "max_abs_diff": self.max_abs_diff,
            "all_reduce_bytes": self.all_reduce_bytes,
            "shard_parameter_bytes": self.shard_parameter_bytes,
            "shard_shapes": [list(shape) for shape in self.shard_shapes],
        }


class TensorParallelPlan:
    def __init__(self, *, world_size: int) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.world_size = world_size

    def run_column_parallel_linear(self, x: Any, weight: Any) -> TensorParallelCaseResult:
        import torch

        values = torch.as_tensor(x).float()
        dense_weight = torch.as_tensor(weight).float()
        if dense_weight.shape[1] % self.world_size != 0:
            raise ValueError("output dimension must be divisible by world_size")
        dense = values @ dense_weight
        shards = torch.chunk(dense_weight, self.world_size, dim=1)
        outputs = [values @ shard for shard in shards]
        gathered = torch.cat(outputs, dim=1)
        diff = (gathered - dense).abs()
        return TensorParallelCaseResult(
            mode="column_parallel",
            world_size=self.world_size,
            input_shape=tuple(values.shape),
            weight_shape=tuple(dense_weight.shape),
            max_abs_diff=float(diff.max().item()),
            all_reduce_bytes=0,
            shard_parameter_bytes=max(_nbytes(shard) for shard in shards),
            shard_shapes=tuple(tuple(shard.shape) for shard in shards),
        )

    def run_row_parallel_linear(self, x: Any, weight: Any) -> TensorParallelCaseResult:
        import torch

        values = torch.as_tensor(x).float()
        dense_weight = torch.as_tensor(weight).float()
        if dense_weight.shape[0] % self.world_size != 0:
            raise ValueError("input dimension must be divisible by world_size")
        dense = values @ dense_weight
        input_shards = torch.chunk(values, self.world_size, dim=1)
        weight_shards = torch.chunk(dense_weight, self.world_size, dim=0)
        partials = [input_shard @ weight_shard for input_shard, weight_shard in zip(input_shards, weight_shards, strict=True)]
        reduced = sum(partials)
        diff = (reduced - dense).abs()
        return TensorParallelCaseResult(
            mode="row_parallel",
            world_size=self.world_size,
            input_shape=tuple(values.shape),
            weight_shape=tuple(dense_weight.shape),
            max_abs_diff=float(diff.max().item()),
            all_reduce_bytes=_nbytes(partials[0]) * max(0, self.world_size - 1),
            shard_parameter_bytes=max(_nbytes(shard) for shard in weight_shards),
            shard_shapes=tuple(tuple(shard.shape) for shard in weight_shards),
        )

    def kv_shard_bytes(self, *, layers: int, kv_heads: int, head_dim: int, tokens: int, dtype_bytes: int = 2) -> int:
        if kv_heads <= 0 or head_dim <= 0 or tokens <= 0 or layers <= 0:
            raise ValueError("KV shard dimensions must be positive")
        heads_per_rank = (kv_heads + self.world_size - 1) // self.world_size
        return layers * 2 * heads_per_rank * tokens * head_dim * dtype_bytes


def _nbytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())

