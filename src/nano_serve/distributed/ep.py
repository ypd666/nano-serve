"""Expert-parallel reference dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExpertDispatchResult:
    world_size: int
    num_experts: int
    token_count: int
    expert_counts: tuple[int, ...]
    rank_counts: tuple[int, ...]
    load_imbalance: int
    all_to_all_bytes: int
    max_abs_diff: float

    def to_dict(self) -> dict[str, object]:
        return {
            "world_size": self.world_size,
            "num_experts": self.num_experts,
            "token_count": self.token_count,
            "expert_counts": list(self.expert_counts),
            "rank_counts": list(self.rank_counts),
            "load_imbalance": self.load_imbalance,
            "all_to_all_bytes": self.all_to_all_bytes,
            "max_abs_diff": self.max_abs_diff,
        }


class ExpertParallelPlan:
    def __init__(self, *, world_size: int, num_experts: int) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.world_size = world_size
        self.num_experts = num_experts

    def expert_rank(self, expert_id: int) -> int:
        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(f"invalid expert id: {expert_id}")
        return expert_id % self.world_size

    def dispatch_and_combine(self, tokens: Any, expert_ids: Any) -> ExpertDispatchResult:
        import torch

        token_values = torch.as_tensor(tokens).float()
        route_ids = torch.as_tensor(expert_ids).long()
        if token_values.shape[0] != route_ids.shape[0]:
            raise ValueError("tokens and expert_ids must have the same first dimension")
        if route_ids.numel() and (
            int(route_ids.min().item()) < 0
            or int(route_ids.max().item()) >= self.num_experts
        ):
            raise ValueError("expert_ids contain out-of-range values")

        expert_counts = [0 for _ in range(self.num_experts)]
        rank_counts = [0 for _ in range(self.world_size)]
        restored = torch.empty_like(token_values)
        for index, expert_id_tensor in enumerate(route_ids):
            expert_id = int(expert_id_tensor.item())
            expert_counts[expert_id] += 1
            rank_counts[self.expert_rank(expert_id)] += 1
            restored[index] = token_values[index]

        diff = (restored - token_values).abs()
        token_bytes = int(token_values.numel() * token_values.element_size())
        return ExpertDispatchResult(
            world_size=self.world_size,
            num_experts=self.num_experts,
            token_count=int(token_values.shape[0]),
            expert_counts=tuple(expert_counts),
            rank_counts=tuple(rank_counts),
            load_imbalance=max(rank_counts) - min(rank_counts) if rank_counts else 0,
            all_to_all_bytes=token_bytes * 2,
            max_abs_diff=float(diff.max().item()) if diff.numel() else 0.0,
        )
