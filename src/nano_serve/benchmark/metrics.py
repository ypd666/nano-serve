"""Benchmark metric dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IterationMetrics:
    iter_id: int
    batch_size: int
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    kv_blocks_used: int | None = None
    kv_blocks_free: int | None = None
    kv_fragmentation: float | None = None
    prefix_cache_hit_tokens: int = 0
    gpu_time_ms: float | None = None
    cpu_schedule_time_ms: float | None = None


@dataclass(frozen=True)
class SystemMetrics:
    output_tokens_per_sec: float | None = None
    total_tokens_per_sec: float | None = None
    requests_per_sec: float | None = None
    goodput_under_slo: float | None = None
    gpu_memory_peak_bytes: int | None = None
    mfu_prefill: float | None = None
    mfu_decode: float | None = None
    sm_active_avg: float | None = None
    hbm_bw_util_avg: float | None = None
    platform: dict[str, object] = field(default_factory=dict)
