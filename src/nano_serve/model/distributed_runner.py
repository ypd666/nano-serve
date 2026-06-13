"""Distributed model runner reference facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nano_serve.distributed import (
    DataParallelRouter,
    ExpertParallelPlan,
    PipelineParallelPlan,
    TensorParallelPlan,
)
from nano_serve.engine.config import EngineConfig


@dataclass(frozen=True)
class DistributedRunResult:
    mode: str
    fields: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, **self.fields}


class DistributedModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        if config.parallel.mode not in {"dp", "tp", "pp", "ep"}:
            raise ValueError(
                "DistributedModelRunner supports dp/tp/pp/ep reference modes"
            )
        self.config = config

    def execute(self, *args: Any, **kwargs: Any) -> DistributedRunResult:
        mode = self.config.parallel.mode
        if mode == "dp":
            request_ids = list(kwargs.get("request_ids", []))
            router = DataParallelRouter(world_size=self.config.parallel.dp_size)
            for request_id in request_ids:
                router.route(str(request_id))
            return DistributedRunResult(mode=mode, fields=router.summary())
        if mode == "tp":
            tp_plan = TensorParallelPlan(world_size=self.config.parallel.tp_size)
            x = kwargs["x"]
            weight = kwargs["weight"]
            tp_result = tp_plan.run_row_parallel_linear(x, weight)
            return DistributedRunResult(mode=mode, fields=tp_result.to_dict())
        if mode == "pp":
            pp_plan = PipelineParallelPlan(stages=self.config.parallel.pp_size)
            pp_result = pp_plan.schedule(microbatches=int(kwargs["microbatches"]))
            return DistributedRunResult(mode=mode, fields=pp_result.to_dict())
        if mode == "ep":
            ep_plan = ExpertParallelPlan(
                world_size=self.config.parallel.ep_size,
                num_experts=int(kwargs["num_experts"]),
            )
            ep_result = ep_plan.dispatch_and_combine(kwargs["tokens"], kwargs["expert_ids"])
            return DistributedRunResult(mode=mode, fields=ep_result.to_dict())
        raise AssertionError(f"unhandled distributed mode: {mode}")

