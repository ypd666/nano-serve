"""Single-node worker lifecycle primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkerState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass(frozen=True)
class WorkerInfo:
    worker_id: str
    rank: int
    world_size: int
    role: str = "model"


@dataclass(frozen=True)
class WorkerLifecycleEvent:
    worker_id: str
    rank: int
    world_size: int
    role: str
    state: str

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "role": self.role,
            "state": self.state,
        }


class Worker:
    def __init__(self, info: WorkerInfo) -> None:
        self.info = info
        self.state = WorkerState.CREATED
        self.events: list[WorkerLifecycleEvent] = [
            self._event(WorkerState.CREATED),
        ]

    def start(self) -> WorkerLifecycleEvent:
        if self.state == WorkerState.RUNNING:
            return self.events[-1]
        if self.state == WorkerState.STOPPED:
            raise RuntimeError("stopped workers cannot be restarted")
        self.state = WorkerState.RUNNING
        event = self._event(self.state)
        self.events.append(event)
        return event

    def stop(self) -> WorkerLifecycleEvent:
        if self.state == WorkerState.STOPPED:
            return self.events[-1]
        self.state = WorkerState.STOPPED
        event = self._event(self.state)
        self.events.append(event)
        return event

    def run(self) -> WorkerLifecycleEvent:
        return self.start()

    def _event(self, state: WorkerState) -> WorkerLifecycleEvent:
        return WorkerLifecycleEvent(
            worker_id=self.info.worker_id,
            rank=self.info.rank,
            world_size=self.info.world_size,
            role=self.info.role,
            state=state.value,
        )

