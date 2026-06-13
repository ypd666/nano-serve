"""Phase 9 prefix-cache benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import EngineConfig
from nano_serve.kv_cache.paged import PagedKVCache
from nano_serve.kv_cache.prefix_cache import PrefixCache, PrefixCacheEntry
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform


@dataclass(frozen=True)
class Phase9PrefixCacheBenchmarkConfig:
    output_dir: Path
    requests: int = 64
    shared_prefix_tokens: int = 512
    unique_suffix_tokens: int = 64
    block_size: int = 16
    cache_blocks: int = 4096
    max_prefix_entries: int | None = None
    prefill_token_time_ms: float = 0.02


def run_phase9_prefix_cache_benchmark(
    config: Phase9PrefixCacheBenchmarkConfig,
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
    engine_config = EngineConfig(
        kv_cache="paged_prefix",
        block_size=config.block_size,
    )
    run_config = {
        "run_id": run_id,
        "phase": "phase9",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "requests": config.requests,
        "shared_prefix_tokens": config.shared_prefix_tokens,
        "unique_suffix_tokens": config.unique_suffix_tokens,
        "block_size": config.block_size,
        "cache_blocks": config.cache_blocks,
        "max_prefix_entries": config.max_prefix_entries,
        "prefill_token_time_ms": config.prefill_token_time_ms,
        "engine_config": engine_config.to_dict(),
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    start_ns = time.monotonic_ns()
    with JSONLEventWriter(events_path) as writer:
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase9"}))
        writer.write(platform_event(platform_info))
        paged_case = _run_case(config, use_prefix_cache=False, writer=writer)
        prefix_case = _run_case(config, use_prefix_cache=True, writer=writer)
        for case in (paged_case, prefix_case):
            writer.write(Event("prefix_cache_case", fields=case))
        end_ns = time.monotonic_ns()
        summary = {
            "run_id": run_id,
            "phase": "phase9",
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": (end_ns - start_ns) / 1_000_000_000,
            "workload": "shared_prefix",
            "scheduler": "single",
            "kv_cache": "paged_prefix",
            "baseline": paged_case,
            "candidate": prefix_case,
            "saved_prefill_tokens": prefix_case["saved_prefill_tokens"],
            "prefix_hit_rate": prefix_case["prefix_hit_rate"],
            "estimated_ttft_improvement_ms": prefix_case["estimated_ttft_improvement_ms"],
            "engine_config": engine_config.to_dict(),
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


def _run_case(
    config: Phase9PrefixCacheBenchmarkConfig,
    *,
    use_prefix_cache: bool,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    cache = PagedKVCache(num_blocks=config.cache_blocks, block_size=config.block_size)
    prefix_cache = (
        PrefixCache(block_size=config.block_size, max_entries=config.max_prefix_entries)
        if use_prefix_cache
        else None
    )
    prompt_tokens = _workload_tokens(config)
    total_input_tokens = 0
    saved_prefill_tokens = 0
    request_ttft_ms: list[float] = []

    for request_index, token_ids in enumerate(prompt_tokens):
        request_id = f"{'prefix' if use_prefix_cache else 'paged'}-{request_index}"
        total_input_tokens += len(token_ids)
        prefix_tokens = 0
        prefix_block_ids: list[int] = []
        if prefix_cache is not None:
            lookup = prefix_cache.lookup(token_ids)
            prefix_tokens = lookup.matched_tokens
            prefix_block_ids = list(lookup.block_ids)
            saved_prefill_tokens += prefix_tokens
            writer.write(
                Event(
                    "prefix_cache_lookup",
                    fields={
                        "request_id": request_id,
                        "case": "paged_prefix",
                        **lookup.to_dict(),
                    },
                )
            )

        if prefix_tokens:
            handle = cache.allocate_prefill_with_prefix(
                request_id,
                len(token_ids),
                prefix_block_ids=prefix_block_ids,
                prefix_tokens=prefix_tokens,
            )
        else:
            handle = cache.allocate_prefill(request_id, len(token_ids))

        if prefix_cache is not None:
            retained_blocks = handle.block_ids[: len(token_ids) // config.block_size]
            insert_result = prefix_cache.insert(
                token_ids,
                retained_blocks,
                on_insert=lambda entry: cache.retain_blocks(list(entry.block_ids)),
                on_evict=lambda entry: _release_evicted(cache, writer, entry),
            )
            writer.write(
                Event(
                    "prefix_cache_insert",
                    fields={
                        "request_id": request_id,
                        "case": "paged_prefix",
                        "token_count": (len(token_ids) // config.block_size)
                        * config.block_size,
                        "block_ids": retained_blocks,
                        **insert_result.to_dict(),
                        **prefix_cache.stats().to_dict(),
                    },
                )
            )

        private_prefill_tokens = len(token_ids) - prefix_tokens
        request_ttft_ms.append(private_prefill_tokens * config.prefill_token_time_ms)
        writer.write(
            Event(
                "prefix_cache_request_end",
                fields={
                    "request_id": request_id,
                    "case": "paged_prefix" if use_prefix_cache else "paged",
                    "input_tokens": len(token_ids),
                    "prefix_hit_tokens": prefix_tokens,
                    "private_prefill_tokens": private_prefill_tokens,
                    "block_ids": handle.block_ids,
                    **cache.stats().to_dict(),
                },
            )
        )

    cache_stats = cache.stats()
    prefix_stats = prefix_cache.stats() if prefix_cache is not None else None
    hit_rate = prefix_stats.hit_rate if prefix_stats is not None else 0.0
    return {
        "case": "paged_prefix" if use_prefix_cache else "paged",
        "requests": config.requests,
        "total_input_tokens": total_input_tokens,
        "saved_prefill_tokens": saved_prefill_tokens,
        "prefix_hit_rate": hit_rate,
        "estimated_ttft_improvement_ms": saved_prefill_tokens
        * config.prefill_token_time_ms,
        "avg_request_ttft_ms": sum(request_ttft_ms) / len(request_ttft_ms),
        "max_request_ttft_ms": max(request_ttft_ms),
        "used_blocks": cache_stats.used_blocks,
        "free_blocks": cache_stats.free_blocks,
        "shared_blocks": cache_stats.shared_blocks,
        "cow_copies": cache_stats.cow_copies,
        "evictions": prefix_stats.evictions if prefix_stats is not None else 0,
        "hit_tokens": prefix_stats.hit_tokens if prefix_stats is not None else 0,
        "cached_block_refs": prefix_stats.cached_block_refs if prefix_stats is not None else 0,
        "cached_entry_tokens": prefix_stats.cached_entry_tokens
        if prefix_stats is not None
        else 0,
    }


def _release_evicted(
    cache: PagedKVCache,
    writer: JSONLEventWriter,
    entry: PrefixCacheEntry,
) -> None:
    cache.release_blocks(list(entry.block_ids))
    writer.write(
        Event(
            "prefix_cache_evict",
            fields={
                "token_count": entry.token_count,
                "block_ids": list(entry.block_ids),
                **cache.stats().to_dict(),
            },
        )
    )


def _workload_tokens(config: Phase9PrefixCacheBenchmarkConfig) -> list[list[int]]:
    shared = list(range(config.shared_prefix_tokens))
    prompts = []
    for request_index in range(config.requests):
        suffix_base = 1_000_000 + request_index * config.unique_suffix_tokens
        suffix = list(range(suffix_base, suffix_base + config.unique_suffix_tokens))
        prompts.append([*shared, *suffix])
    return prompts


def _validate_config(config: Phase9PrefixCacheBenchmarkConfig) -> None:
    if config.requests <= 0:
        raise ValueError("requests must be positive")
    if config.shared_prefix_tokens <= 0:
        raise ValueError("shared_prefix_tokens must be positive")
    if config.unique_suffix_tokens < 0:
        raise ValueError("unique_suffix_tokens must be non-negative")
    if config.block_size <= 0:
        raise ValueError("block_size must be positive")
    if config.cache_blocks <= 0:
        raise ValueError("cache_blocks must be positive")
    if config.max_prefix_entries is not None and config.max_prefix_entries <= 0:
        raise ValueError("max_prefix_entries must be positive")
    if config.prefill_token_time_ms <= 0:
        raise ValueError("prefill_token_time_ms must be positive")


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
