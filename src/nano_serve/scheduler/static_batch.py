"""Static batch scheduler for fixed-shape offline batches."""

from __future__ import annotations

from dataclasses import dataclass

from nano_serve.engine.batch import BatchKind, BatchPlan
from nano_serve.engine.request import RequestState
from nano_serve.scheduler.base import ScheduleBudget


@dataclass(frozen=True)
class StaticBatchWaste:
    batch_size: int
    active_slots: int
    inactive_slots: int
    max_tokens_per_slot: int
    real_tokens: int
    padded_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "active_slots": self.active_slots,
            "inactive_slots": self.inactive_slots,
            "max_tokens_per_slot": self.max_tokens_per_slot,
            "real_tokens": self.real_tokens,
            "padded_tokens": self.padded_tokens,
        }


class StaticBatchScheduler:
    def schedule(
        self,
        waiting: list[RequestState],
        running: list[RequestState],
        kv_cache: object,
        budget: ScheduleBudget,
    ) -> BatchPlan:
        del kv_cache
        candidates = running or waiting[: budget.max_num_seqs]
        if not candidates:
            return BatchPlan(kind=BatchKind.DECODE, request_ids=[], input_token_ids=[])
        kind = BatchKind.DECODE if running else BatchKind.PREFILL
        input_token_ids = [
            [*request.prompt_token_ids, *request.output_token_ids]
            if kind == BatchKind.DECODE
            else request.prompt_token_ids
            for request in candidates
        ]
        return BatchPlan(
            kind=kind,
            request_ids=[request.request_id for request in candidates],
            input_token_ids=input_token_ids,
            num_prefill_tokens=sum(len(request.prompt_token_ids) for request in candidates)
            if kind == BatchKind.PREFILL
            else 0,
            num_decode_tokens=sum(0 if request.is_terminal else 1 for request in candidates)
            if kind == BatchKind.DECODE
            else 0,
            metadata=static_waste(candidates).to_dict(),
        )


def static_waste(requests: list[RequestState]) -> StaticBatchWaste:
    lengths = [len(request.prompt_token_ids) + len(request.output_token_ids) for request in requests]
    max_tokens = max(lengths, default=0)
    real_tokens = sum(lengths)
    active_slots = sum(0 if request.is_terminal else 1 for request in requests)
    batch_size = len(requests)
    padded_tokens = batch_size * max_tokens - real_tokens
    return StaticBatchWaste(
        batch_size=batch_size,
        active_slots=active_slots,
        inactive_slots=batch_size - active_slots,
        max_tokens_per_slot=max_tokens,
        real_tokens=real_tokens,
        padded_tokens=padded_tokens,
    )
