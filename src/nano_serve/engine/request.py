"""Request state and request-level metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from nano_serve.sampling.base import SamplingParams


class RequestStatus(StrEnum):
    WAITING = "WAITING"
    PREFILL = "PREFILL"
    DECODE = "DECODE"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"


@dataclass
class RequestMetrics:
    arrival_time_ns: int
    first_scheduled_time_ns: int | None = None
    prefill_start_time_ns: int | None = None
    prefill_end_time_ns: int | None = None
    first_token_time_ns: int | None = None
    last_token_time_ns: int | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_time_ns is None:
            return None
        return (self.first_token_time_ns - self.arrival_time_ns) / 1_000_000

    def tpot_ms(self, output_tokens: int) -> float | None:
        if output_tokens <= 1:
            return None
        if self.first_token_time_ns is None or self.last_token_time_ns is None:
            return None
        return (self.last_token_time_ns - self.first_token_time_ns) / (
            1_000_000 * (output_tokens - 1)
        )

    @property
    def e2e_ms(self) -> float | None:
        if self.last_token_time_ns is None:
            return None
        return (self.last_token_time_ns - self.arrival_time_ns) / 1_000_000


@dataclass
class RequestState:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    metrics: RequestMetrics
    output_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    prefill_cursor: int = 0
    max_new_tokens: int = 16
    stop_reason: str | None = None
    kv_handle: object | None = None
    block_table: list[int] = field(default_factory=list)
    phase_metadata: list[dict[str, object]] = field(default_factory=list)
    prefix_cache_hit_tokens: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_terminal(self) -> bool:
        return self.status in {RequestStatus.FINISHED, RequestStatus.ABORTED}

