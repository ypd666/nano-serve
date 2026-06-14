"""Phase 5 paged KV allocator benchmark."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nano_serve.benchmark.profiler import nvtx_label, nvtx_range
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.kv_cache.paged import PagedKVCache
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform


@dataclass(frozen=True)
class Phase5KVBenchmarkConfig:
    output_dir: Path
    num_blocks: int = 128
    block_size: int = 16
    num_requests: int = 64
    max_prefill_tokens: int = 128
    max_decode_tokens: int = 64
    seed: int = 0
    enable_nvtx: bool = False


def run_phase5_kv_benchmark(config: Phase5KVBenchmarkConfig) -> dict[str, object]:
    _validate_config(config)
    platform_info = detect_platform()
    run_id = _run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"

    run_config = {
        "run_id": run_id,
        "phase": "phase5",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "num_blocks": config.num_blocks,
        "block_size": config.block_size,
        "num_requests": config.num_requests,
        "max_prefill_tokens": config.max_prefill_tokens,
        "max_decode_tokens": config.max_decode_tokens,
        "seed": config.seed,
        "enable_nvtx": config.enable_nvtx,
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    rng = random.Random(config.seed)
    cache = PagedKVCache(num_blocks=config.num_blocks, block_size=config.block_size)
    request_ids = [f"req-{index}" for index in range(config.num_requests)]
    prefill_tokens = {
        request_id: rng.randint(1, config.max_prefill_tokens)
        for request_id in request_ids
    }
    decode_tokens = {
        request_id: rng.randint(0, config.max_decode_tokens)
        for request_id in request_ids
    }
    start_ns = time.monotonic_ns()
    allocation_failures = 0
    decode_failures = 0
    allocated_requests: list[str] = []

    with (
        JSONLEventWriter(events_path) as writer,
        nvtx_range(nvtx_label("phase5", "run"), enabled=config.enable_nvtx),
    ):
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase5"}))
        writer.write(platform_event(platform_info))

        for request_id in request_ids:
            with nvtx_range(
                nvtx_label(
                    "phase5",
                    "prefill",
                    request_id=request_id,
                    tokens=prefill_tokens[request_id],
                ),
                enabled=config.enable_nvtx,
            ):
                try:
                    handle = cache.allocate_prefill(request_id, prefill_tokens[request_id])
                except MemoryError as exc:
                    allocation_failures += 1
                    writer.write(
                        Event(
                            "paged_kv_oom",
                            fields={
                                "request_id": request_id,
                                "phase": "prefill",
                                "message": str(exc),
                                **cache.stats().to_dict(),
                            },
                        )
                    )
                    continue
            allocated_requests.append(request_id)
            writer.write(
                Event(
                    "paged_kv_prefill",
                    fields={
                        "request_id": request_id,
                        "tokens": prefill_tokens[request_id],
                        "block_ids": handle.block_ids,
                        **cache.stats().to_dict(),
                    },
                )
            )

            with nvtx_range(
                nvtx_label(
                    "phase5",
                    "decode",
                    request_id=request_id,
                    tokens=decode_tokens[request_id],
                ),
                enabled=config.enable_nvtx,
            ):
                for _ in range(decode_tokens[request_id]):
                    try:
                        handle = cache.allocate_decode_slot(request_id)
                    except MemoryError as exc:
                        decode_failures += 1
                        writer.write(
                            Event(
                                "paged_kv_oom",
                                fields={
                                    "request_id": request_id,
                                    "phase": "decode",
                                    "message": str(exc),
                                    **cache.stats().to_dict(),
                                },
                            )
                        )
                        break
                writer.write(
                    Event(
                        "paged_kv_decode_end",
                        fields={
                            "request_id": request_id,
                            "target_decode_tokens": decode_tokens[request_id],
                            "sequence_length": handle.num_tokens,
                            "block_ids": handle.block_ids,
                            **cache.stats().to_dict(),
                        },
                    )
                )

        peak_stats = cache.stats()
        for request_id in allocated_requests[::2]:
            with nvtx_range(
                nvtx_label("phase5", "free", request_id=request_id),
                enabled=config.enable_nvtx,
            ):
                cache.free(request_id)
            writer.write(
                Event(
                    "paged_kv_free",
                    fields={"request_id": request_id, **cache.stats().to_dict()},
                )
            )
        end_ns = time.monotonic_ns()
        final_stats = cache.stats()
        summary = {
            "run_id": run_id,
            "phase": "phase5",
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": (end_ns - start_ns) / 1_000_000_000,
            "num_blocks": config.num_blocks,
            "block_size": config.block_size,
            "num_requests": config.num_requests,
            "allocated_requests": len(allocated_requests),
            "allocation_failures": allocation_failures,
            "decode_failures": decode_failures,
            "peak": peak_stats.to_dict(),
            "final": final_stats.to_dict(),
            "platform": platform_info.to_dict(),
            "artifacts": {
                "run_config": str(run_config_path),
                "events": str(events_path),
                "summary": str(summary_path),
                "report": str(report_path),
            },
        }
        writer.write(Event("run_end", fields=summary))

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report_path, summary)
    return summary


def _validate_config(config: Phase5KVBenchmarkConfig) -> None:
    if config.num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if config.block_size <= 0:
        raise ValueError("block_size must be positive")
    if config.num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if config.max_prefill_tokens <= 0:
        raise ValueError("max_prefill_tokens must be positive")
    if config.max_decode_tokens < 0:
        raise ValueError("max_decode_tokens must be non-negative")


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
