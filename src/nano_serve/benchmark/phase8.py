"""Phase 8 chunked-prefill scheduler benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import EngineConfig
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform


@dataclass(frozen=True)
class Phase8ChunkedPrefillBenchmarkConfig:
    output_dir: Path
    chunk_sizes: tuple[int, ...] = (128, 512, 2048)
    long_prompt_tokens: int = 8192
    decode_requests: int = 8
    decode_tokens_per_request: int = 128
    max_num_seqs: int = 9
    max_num_batched_tokens: int = 4096
    prefill_token_time_ms: float = 0.02
    decode_token_time_ms: float = 0.05


@dataclass
class _SimRequest:
    request_id: str
    prompt_tokens: int
    remaining_decode_tokens: int
    prefill_cursor: int = 0
    first_token_ms: float | None = None
    last_token_ms: float | None = None
    decode_intervals_ms: list[float] | None = None
    last_decode_ms: float | None = None

    @property
    def prefill_done(self) -> bool:
        return self.prefill_cursor >= self.prompt_tokens

    @property
    def finished(self) -> bool:
        return self.prefill_done and self.remaining_decode_tokens <= 0


def run_phase8_chunked_prefill_benchmark(
    config: Phase8ChunkedPrefillBenchmarkConfig,
) -> dict[str, object]:
    _validate_config(config)
    platform_info = detect_platform()
    run_id = _run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    plot_path = run_dir / "chunk_size_frontier.png"
    engine_config = EngineConfig(
        scheduler="chunked_prefill",
        kv_cache="contiguous",
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_prefill_chunk_tokens=config.chunk_sizes[0],
    )

    chunk_sizes = _normalized_chunk_sizes(config)
    run_config = {
        "run_id": run_id,
        "phase": "phase8",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "chunk_sizes": list(chunk_sizes),
        "long_prompt_tokens": config.long_prompt_tokens,
        "decode_requests": config.decode_requests,
        "decode_tokens_per_request": config.decode_tokens_per_request,
        "max_num_seqs": config.max_num_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "prefill_token_time_ms": config.prefill_token_time_ms,
        "decode_token_time_ms": config.decode_token_time_ms,
        "engine_config": engine_config.to_dict(),
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    cases: list[dict[str, object]] = []
    start_ns = time.monotonic_ns()
    with JSONLEventWriter(events_path) as writer:
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase8"}))
        writer.write(platform_event(platform_info))
        for chunk_size in chunk_sizes:
            case = _simulate_case(config, chunk_size=chunk_size, writer=writer)
            cases.append(case)
            writer.write(Event("chunked_prefill_case", fields=case))
        _write_frontier_plot(plot_path, cases)
        end_ns = time.monotonic_ns()
        summary = {
            "run_id": run_id,
            "phase": "phase8",
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": (end_ns - start_ns) / 1_000_000_000,
            "scheduler": "chunked_prefill",
            "kv_cache": "contiguous",
            "workload": "long_prefill_interference",
            "chunk_sizes": list(chunk_sizes),
            "cases": cases,
            "best_decode_tpot_case": min(
                cases,
                key=lambda item: _float_metric(item, "decode_tpot_p90_ms"),
            ),
            "best_decode_gap_case": min(
                cases,
                key=lambda item: _float_metric(item, "decode_tpot_max_ms"),
            ),
            "best_long_ttft_case": min(
                cases,
                key=lambda item: _float_metric(item, "long_ttft_ms"),
            ),
            "engine_config": engine_config.to_dict(),
            "platform": platform_info.to_dict(),
            "artifacts": {
                "run_config": str(run_config_path),
                "events": str(events_path),
                "summary": str(summary_path),
                "report": str(report_path),
                "frontier_plot": str(plot_path) if plot_path.exists() else None,
            },
        }
        writer.write(Event("run_end", fields=summary))

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report_path, summary)
    return summary


def _simulate_case(
    config: Phase8ChunkedPrefillBenchmarkConfig,
    *,
    chunk_size: int,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    decode_requests = [
        _SimRequest(
            request_id=f"decode-{index}",
            prompt_tokens=1,
            prefill_cursor=1,
            remaining_decode_tokens=config.decode_tokens_per_request,
            decode_intervals_ms=[],
            first_token_ms=0.0,
            last_token_ms=0.0,
            last_decode_ms=0.0,
        )
        for index in range(config.decode_requests)
    ]
    long_request = _SimRequest(
        request_id="long-prefill",
        prompt_tokens=config.long_prompt_tokens,
        remaining_decode_tokens=1,
        decode_intervals_ms=[],
    )
    running = [*decode_requests, long_request]
    now_ms = 0.0
    iteration = 0
    mixed_iterations = 0
    prefill_chunks = 0
    decode_stall_ms = 0.0
    no_chunk = chunk_size >= config.long_prompt_tokens

    while any(not request.finished for request in running):
        active_decode = [
            request
            for request in running
            if request.prefill_done and request.remaining_decode_tokens > 0
        ]
        prefill_request = next(
            (
                request
                for request in running
                if not request.prefill_done and request.request_id == long_request.request_id
            ),
            None,
        )
        selected_decode = list(active_decode)
        token_budget = max(0, config.max_num_batched_tokens - len(selected_decode))
        chunk_tokens = 0
        if prefill_request is not None and token_budget > 0:
            chunk_tokens = min(
                chunk_size,
                token_budget,
                prefill_request.prompt_tokens - prefill_request.prefill_cursor,
            )
        if no_chunk and prefill_request is not None and prefill_request.prefill_cursor == 0:
            selected_decode = []
            chunk_tokens = prefill_request.prompt_tokens

        batch_kind = _batch_kind(decode_tokens=len(selected_decode), prefill_tokens=chunk_tokens)
        if batch_kind == "MIXED":
            mixed_iterations += 1
        if chunk_tokens:
            prefill_chunks += 1

        writer.write(
            Event(
                "chunked_prefill_iteration_start",
                fields={
                    "case_chunk_size": chunk_size,
                    "iteration": iteration,
                    "batch_kind": batch_kind,
                    "decode_tokens": len(selected_decode),
                    "prefill_tokens": chunk_tokens,
                    "running_requests": sum(not request.finished for request in running),
                    "long_prefill_cursor": long_request.prefill_cursor,
                },
            )
        )

        elapsed_ms = (
            len(selected_decode) * config.decode_token_time_ms
            + chunk_tokens * config.prefill_token_time_ms
        )
        now_ms += elapsed_ms

        for request in selected_decode:
            if request.last_decode_ms is not None:
                interval = now_ms - request.last_decode_ms
                request.decode_intervals_ms = request.decode_intervals_ms or []
                request.decode_intervals_ms.append(interval)
                expected_decode_ms = config.decode_token_time_ms
                decode_stall_ms += max(0.0, interval - expected_decode_ms)
            request.remaining_decode_tokens -= 1
            if request.first_token_ms is None:
                request.first_token_ms = now_ms
            request.last_token_ms = now_ms
            request.last_decode_ms = now_ms

        if prefill_request is not None and chunk_tokens:
            prefill_request.prefill_cursor += chunk_tokens
            if prefill_request.prefill_done:
                prefill_request.first_token_ms = now_ms

        writer.write(
            Event(
                "chunked_prefill_iteration_end",
                fields={
                    "case_chunk_size": chunk_size,
                    "iteration": iteration,
                    "batch_kind": batch_kind,
                    "decode_tokens": len(selected_decode),
                    "prefill_tokens": chunk_tokens,
                    "running_requests": sum(not request.finished for request in running),
                    "long_prefill_cursor": long_request.prefill_cursor,
                    "simulated_time_ms": now_ms,
                    "decode_stall_ms": decode_stall_ms,
                },
            )
        )
        iteration += 1

    intervals = [
        interval
        for request in decode_requests
        for interval in (request.decode_intervals_ms or [])
    ]
    return {
        "chunk_size": chunk_size,
        "baseline_no_chunk": no_chunk,
        "iterations": iteration,
        "mixed_iterations": mixed_iterations,
        "prefill_chunks": prefill_chunks,
        "long_prompt_tokens": config.long_prompt_tokens,
        "long_ttft_ms": long_request.first_token_ms,
        "decode_tpot_p50_ms": _percentile(intervals, 0.50),
        "decode_tpot_p90_ms": _percentile(intervals, 0.90),
        "decode_tpot_p99_ms": _percentile(intervals, 0.99),
        "decode_tpot_max_ms": max(intervals, default=None),
        "decode_stall_ms": decode_stall_ms,
        "simulated_e2e_ms": now_ms,
    }


def _write_frontier_plot(path: Path, cases: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x = [_int_metric(case, "chunk_size") for case in cases]
    ttft = [_float_metric(case, "long_ttft_ms") for case in cases]
    tpot = [_float_metric(case, "decode_tpot_p90_ms") for case in cases]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(x, ttft, marker="o", label="long TTFT ms")
    axis.plot(x, tpot, marker="s", label="decode TPOT p90 ms")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("max prefill chunk tokens")
    axis.set_ylabel("simulated latency ms")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _normalized_chunk_sizes(config: Phase8ChunkedPrefillBenchmarkConfig) -> tuple[int, ...]:
    sizes = sorted(set((*config.chunk_sizes, config.long_prompt_tokens)))
    return tuple(size for size in sizes if size > 0)


def _batch_kind(*, decode_tokens: int, prefill_tokens: int) -> str:
    if decode_tokens and prefill_tokens:
        return "MIXED"
    if prefill_tokens:
        return "PREFILL"
    return "DECODE"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _float_metric(case: dict[str, Any], name: str) -> float:
    value = case[name]
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _int_metric(case: dict[str, Any], name: str) -> int:
    value = case[name]
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_config(config: Phase8ChunkedPrefillBenchmarkConfig) -> None:
    if not config.chunk_sizes:
        raise ValueError("chunk_sizes must not be empty")
    if any(size <= 0 for size in config.chunk_sizes):
        raise ValueError("chunk_sizes must be positive")
    if config.long_prompt_tokens <= 0:
        raise ValueError("long_prompt_tokens must be positive")
    if config.decode_requests <= 0:
        raise ValueError("decode_requests must be positive")
    if config.decode_tokens_per_request <= 0:
        raise ValueError("decode_tokens_per_request must be positive")
    if config.max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    if config.max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")
    if config.prefill_token_time_ms <= 0:
        raise ValueError("prefill_token_time_ms must be positive")
    if config.decode_token_time_ms <= 0:
        raise ValueError("decode_token_time_ms must be positive")


def _run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()
