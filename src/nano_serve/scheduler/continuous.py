"""Continuous batching scheduler for iteration-level decode."""

from __future__ import annotations

import time
from dataclasses import dataclass

from nano_serve.engine.batch import BatchKind, BatchPlan
from nano_serve.engine.request import RequestState
from nano_serve.scheduler.base import ScheduleBudget
from nano_serve.scheduler.policies import SchedulerPolicy


@dataclass(frozen=True)
class ContinuousScheduleStats:
    policy: str
    running_count: int
    waiting_count: int
    admitted_count: int
    selected_count: int
    cpu_schedule_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "running_count": self.running_count,
            "waiting_count": self.waiting_count,
            "admitted_count": self.admitted_count,
            "selected_count": self.selected_count,
            "cpu_schedule_time_ms": self.cpu_schedule_time_ms,
        }


class ContinuousScheduler:
    def __init__(self, policy: SchedulerPolicy = SchedulerPolicy.FCFS) -> None:
        self.policy = policy

    def schedule(
        self,
        waiting: list[RequestState],
        running: list[RequestState],
        kv_cache: object,
        budget: ScheduleBudget,
    ) -> BatchPlan:
        del kv_cache
        start_ns = time.monotonic_ns()
        admitted = _admit_fcfs(waiting, running, budget.max_num_seqs)
        candidates = _select_candidates(waiting, running, budget, self.policy)
        kind = _batch_kind(candidates)
        input_token_ids = [
            [*request.prompt_token_ids, *request.output_token_ids] for request in candidates
        ]
        num_prefill_tokens = sum(
            len(request.prompt_token_ids)
            for request in candidates
            if request.num_output_tokens == 0
        )
        num_decode_tokens = sum(
            1 for request in candidates if request.num_output_tokens > 0
        )
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        stats = ContinuousScheduleStats(
            policy=self.policy.value,
            running_count=len(running),
            waiting_count=len(waiting),
            admitted_count=admitted,
            selected_count=len(candidates),
            cpu_schedule_time_ms=elapsed_ms,
        )
        return BatchPlan(
            kind=kind,
            request_ids=[request.request_id for request in candidates],
            input_token_ids=input_token_ids,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            metadata=stats.to_dict(),
        )


def _admit_fcfs(
    waiting: list[RequestState],
    running: list[RequestState],
    max_num_seqs: int,
) -> int:
    admitted = 0
    while waiting and len(running) < max_num_seqs:
        running.append(waiting.pop(0))
        admitted += 1
    return admitted


def _select_candidates(
    waiting: list[RequestState],
    running: list[RequestState],
    budget: ScheduleBudget,
    policy: SchedulerPolicy,
) -> list[RequestState]:
    del waiting
    active = [request for request in running if not request.is_terminal]
    if policy == SchedulerPolicy.DECODE_FIRST:
        active.sort(key=lambda request: (request.num_output_tokens == 0, request.request_id))
    elif policy == SchedulerPolicy.PREFILL_FIRST:
        active.sort(key=lambda request: (request.num_output_tokens > 0, request.request_id))
    else:
        active.sort(key=lambda request: request.metrics.arrival_time_ns)
    return _fit_token_budget(active, budget.max_num_batched_tokens)


def _fit_token_budget(
    candidates: list[RequestState],
    max_num_batched_tokens: int,
) -> list[RequestState]:
    selected: list[RequestState] = []
    used_tokens = 0
    for request in candidates:
        context_len = request.num_prompt_tokens + request.num_output_tokens
        token_cost = max(1, context_len)
        if selected and used_tokens + token_cost > max_num_batched_tokens:
            break
        selected.append(request)
        used_tokens += token_cost
    return selected


def _batch_kind(candidates: list[RequestState]) -> BatchKind:
    has_prefill = any(request.num_output_tokens == 0 for request in candidates)
    has_decode = any(request.num_output_tokens > 0 for request in candidates)
    if has_prefill and has_decode:
        return BatchKind.MIXED
    if has_prefill:
        return BatchKind.PREFILL
    return BatchKind.DECODE
