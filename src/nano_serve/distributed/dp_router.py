"""Data-parallel routing reference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataParallelAssignment:
    request_id: str
    replica_rank: int
    queue_depth_before: int

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "replica_rank": self.replica_rank,
            "queue_depth_before": self.queue_depth_before,
        }


class DataParallelRouter:
    def __init__(self, *, world_size: int, policy: str = "least_loaded") -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if policy not in {"least_loaded", "round_robin"}:
            raise ValueError(f"unsupported DP policy: {policy}")
        self.world_size = world_size
        self.policy = policy
        self._next_rank = 0
        self._queue_depths = [0 for _ in range(world_size)]
        self.assignments: list[DataParallelAssignment] = []

    @property
    def queue_depths(self) -> tuple[int, ...]:
        return tuple(self._queue_depths)

    def route(self, request_id: str) -> DataParallelAssignment:
        if self.policy == "round_robin":
            rank = self._next_rank
            self._next_rank = (self._next_rank + 1) % self.world_size
        else:
            rank = min(range(self.world_size), key=lambda item: self._queue_depths[item])
        assignment = DataParallelAssignment(
            request_id=request_id,
            replica_rank=rank,
            queue_depth_before=self._queue_depths[rank],
        )
        self._queue_depths[rank] += 1
        self.assignments.append(assignment)
        return assignment

    def complete(self, replica_rank: int) -> None:
        if replica_rank < 0 or replica_rank >= self.world_size:
            raise ValueError(f"invalid replica rank: {replica_rank}")
        self._queue_depths[replica_rank] = max(0, self._queue_depths[replica_rank] - 1)

    def summary(self) -> dict[str, object]:
        counts = [0 for _ in range(self.world_size)]
        for assignment in self.assignments:
            counts[assignment.replica_rank] += 1
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        return {
            "policy": self.policy,
            "world_size": self.world_size,
            "assignments": [item.to_dict() for item in self.assignments],
            "assignment_counts": counts,
            "imbalance": max_count - min_count,
            "final_queue_depths": list(self.queue_depths),
        }

