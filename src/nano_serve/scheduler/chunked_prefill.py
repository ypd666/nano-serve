"""Chunked prefill scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass

from nano_serve.engine.batch import BatchKind, BatchPlan
from nano_serve.engine.request import RequestState, RequestStatus
from nano_serve.scheduler.base import ScheduleBudget


@dataclass(frozen=True)
class PrefillChunk:
    request_id: str
    start: int
    end: int

    @property
    def num_tokens(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "start": self.start,
            "end": self.end,
            "num_tokens": self.num_tokens,
        }


@dataclass(frozen=True)
class ChunkedPrefillScheduleStats:
    running_count: int
    waiting_count: int
    admitted_count: int
    selected_count: int
    decode_count: int
    prefill_count: int
    prefill_tokens: int
    decode_tokens: int
    max_prefill_chunk_tokens: int
    cpu_schedule_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "running_count": self.running_count,
            "waiting_count": self.waiting_count,
            "admitted_count": self.admitted_count,
            "selected_count": self.selected_count,
            "decode_count": self.decode_count,
            "prefill_count": self.prefill_count,
            "num_prefill_tokens": self.prefill_tokens,
            "num_decode_tokens": self.decode_tokens,
            "max_prefill_chunk_tokens": self.max_prefill_chunk_tokens,
            "cpu_schedule_time_ms": self.cpu_schedule_time_ms,
        }


class ChunkedPrefillScheduler:
    """Decode-maximal scheduler with bounded prompt chunks."""

    def schedule(
        self,
        waiting: list[RequestState],
        running: list[RequestState],
        kv_cache: object,
        budget: ScheduleBudget,
    ) -> BatchPlan:
        del kv_cache
        if budget.max_prefill_tokens is None or budget.max_prefill_tokens <= 0:
            raise ValueError("chunked prefill requires a positive max_prefill_tokens budget")
        start_ns = time.monotonic_ns()
        admitted = _admit_fcfs(waiting, running, budget.max_num_seqs)
        selected: list[RequestState] = []
        input_token_ids: list[list[int]] = []
        prefill_chunks: list[PrefillChunk] = []
        used_tokens = 0

        decode_candidates = sorted(
            (
                request
                for request in running
                if not request.is_terminal
                and request.status == RequestStatus.DECODE
                and request.prefill_cursor >= request.num_prompt_tokens
            ),
            key=lambda request: request.metrics.arrival_time_ns,
        )
        for request in decode_candidates:
            if selected and used_tokens + 1 > budget.max_num_batched_tokens:
                break
            selected.append(request)
            input_token_ids.append([*request.prompt_token_ids, *request.output_token_ids])
            used_tokens += 1

        prefill_candidates = sorted(
            (
                request
                for request in running
                if not request.is_terminal and request.prefill_cursor < request.num_prompt_tokens
            ),
            key=lambda request: request.metrics.arrival_time_ns,
        )
        for request in prefill_candidates:
            remaining_budget = budget.max_num_batched_tokens - used_tokens
            if remaining_budget <= 0:
                break
            chunk_tokens = min(
                budget.max_prefill_tokens,
                remaining_budget,
                request.num_prompt_tokens - request.prefill_cursor,
            )
            if chunk_tokens <= 0:
                continue
            start = request.prefill_cursor
            end = start + chunk_tokens
            selected.append(request)
            input_token_ids.append(request.prompt_token_ids[start:end])
            prefill_chunks.append(PrefillChunk(request.request_id, start, end))
            used_tokens += chunk_tokens

        decode_count = sum(1 for request in selected if request.prefill_cursor >= request.num_prompt_tokens)
        prefill_tokens = sum(chunk.num_tokens for chunk in prefill_chunks)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        stats = ChunkedPrefillScheduleStats(
            running_count=len(running),
            waiting_count=len(waiting),
            admitted_count=admitted,
            selected_count=len(selected),
            decode_count=decode_count,
            prefill_count=len(prefill_chunks),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_count,
            max_prefill_chunk_tokens=budget.max_prefill_tokens,
            cpu_schedule_time_ms=elapsed_ms,
        )
        return BatchPlan(
            kind=_batch_kind(decode_count=decode_count, prefill_count=len(prefill_chunks)),
            request_ids=[request.request_id for request in selected],
            input_token_ids=input_token_ids,
            num_prefill_tokens=prefill_tokens,
            num_decode_tokens=decode_count,
            metadata={
                **stats.to_dict(),
                "prefill_chunks": [chunk.to_dict() for chunk in prefill_chunks],
            },
        )


def _admit_fcfs(
    waiting: list[RequestState],
    running: list[RequestState],
    max_num_seqs: int,
) -> int:
    admitted = 0
    while waiting and len(running) < max_num_seqs:
        request = waiting.pop(0)
        request.status = RequestStatus.PREFILL
        running.append(request)
        admitted += 1
    return admitted


def _batch_kind(*, decode_count: int, prefill_count: int) -> BatchKind:
    if decode_count and prefill_count:
        return BatchKind.MIXED
    if prefill_count:
        return BatchKind.PREFILL
    return BatchKind.DECODE
