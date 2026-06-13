"""Pipeline-parallel reference schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStageEvent:
    microbatch: int
    stage: int
    start_step: int
    end_step: int

    def to_dict(self) -> dict[str, object]:
        return {
            "microbatch": self.microbatch,
            "stage": self.stage,
            "start_step": self.start_step,
            "end_step": self.end_step,
        }


@dataclass(frozen=True)
class PipelineScheduleResult:
    stages: int
    microbatches: int
    events: tuple[PipelineStageEvent, ...]
    total_steps: int
    useful_slots: int
    total_slots: int
    bubble_slots: int
    bubble_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "stages": self.stages,
            "microbatches": self.microbatches,
            "events": [event.to_dict() for event in self.events],
            "total_steps": self.total_steps,
            "useful_slots": self.useful_slots,
            "total_slots": self.total_slots,
            "bubble_slots": self.bubble_slots,
            "bubble_ratio": self.bubble_ratio,
        }


class PipelineParallelPlan:
    def __init__(self, *, stages: int) -> None:
        if stages <= 0:
            raise ValueError("stages must be positive")
        self.stages = stages

    def schedule(self, *, microbatches: int) -> PipelineScheduleResult:
        if microbatches <= 0:
            raise ValueError("microbatches must be positive")
        events: list[PipelineStageEvent] = []
        for microbatch in range(microbatches):
            for stage in range(self.stages):
                start = microbatch + stage
                events.append(
                    PipelineStageEvent(
                        microbatch=microbatch,
                        stage=stage,
                        start_step=start,
                        end_step=start + 1,
                    )
                )
        total_steps = microbatches + self.stages - 1
        useful_slots = microbatches * self.stages
        total_slots = total_steps * self.stages
        bubble_slots = total_slots - useful_slots
        return PipelineScheduleResult(
            stages=self.stages,
            microbatches=microbatches,
            events=tuple(events),
            total_steps=total_steps,
            useful_slots=useful_slots,
            total_slots=total_slots,
            bubble_slots=bubble_slots,
            bubble_ratio=bubble_slots / total_slots if total_slots else 0.0,
        )

