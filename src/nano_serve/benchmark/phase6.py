"""Phase 6 torch gather paged-attention benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nano_serve.attention import TorchGatherPagedAttention
from nano_serve.benchmark.profiler import nvtx_label, nvtx_range
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import EngineConfig
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform


@dataclass(frozen=True)
class Phase6PagedAttentionBenchmarkConfig:
    output_dir: Path
    batch_size: int = 2
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    context_lens: tuple[int, ...] = (128, 512, 1024)
    block_sizes: tuple[int, ...] = (8, 16, 32)
    repeats: int = 5
    seed: int = 0
    enable_nvtx: bool = False


def run_phase6_paged_attention_benchmark(
    config: Phase6PagedAttentionBenchmarkConfig,
) -> dict[str, object]:
    _validate_config(config)
    import torch

    platform_info = detect_platform()
    run_id = _run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    engine_config = EngineConfig(
        kv_cache="paged",
        attention_backend="torch_gather_paged",
        block_size=config.block_sizes[0],
    )

    run_config = {
        "run_id": run_id,
        "phase": "phase6",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "batch_size": config.batch_size,
        "query_heads": config.query_heads,
        "kv_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "context_lens": list(config.context_lens),
        "block_sizes": list(config.block_sizes),
        "repeats": config.repeats,
        "seed": config.seed,
        "enable_nvtx": config.enable_nvtx,
        "device": str(device),
        "dtype": str(dtype),
        "engine_config": engine_config.to_dict(),
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    attention = TorchGatherPagedAttention()
    results: list[dict[str, object]] = []
    start_ns = time.monotonic_ns()
    with (
        JSONLEventWriter(events_path) as writer,
        nvtx_range(nvtx_label("phase6", "run"), enabled=config.enable_nvtx),
    ):
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase6"}))
        writer.write(platform_event(platform_info))
        for context_len in config.context_lens:
            for block_size in config.block_sizes:
                with nvtx_range(
                    nvtx_label(
                        "phase6",
                        "case",
                        block_size=block_size,
                        context_len=context_len,
                    ),
                    enabled=config.enable_nvtx,
                ):
                    case = _run_case(
                        attention,
                        batch_size=config.batch_size,
                        query_heads=config.query_heads,
                        kv_heads=config.kv_heads,
                        head_dim=config.head_dim,
                        context_len=context_len,
                        block_size=block_size,
                        repeats=config.repeats,
                        seed=config.seed,
                        device=device,
                        dtype=dtype,
                        enable_nvtx=config.enable_nvtx,
                    )
                results.append(case)
                writer.write(Event("paged_attention_case", fields=case))
        end_ns = time.monotonic_ns()
        summary = {
            "run_id": run_id,
            "phase": "phase6",
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": (end_ns - start_ns) / 1_000_000_000,
            "device": str(device),
            "dtype": str(dtype),
            "cases": results,
            "max_gather_time_ms": max(
                (_float_metric(case, "gather_time_ms") for case in results),
                default=0.0,
            ),
            "max_attention_time_ms": max(
                (_float_metric(case, "attention_time_ms") for case in results),
                default=0.0,
            ),
            "max_abs_diff": max(
                (_float_metric(case, "max_abs_diff") for case in results),
                default=0.0,
            ),
            "max_gather_temp_bytes": max(
                (_int_metric(case, "gather_temp_bytes") for case in results),
                default=0,
            ),
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
    attention: TorchGatherPagedAttention,
    *,
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    context_len: int,
    block_size: int,
    repeats: int,
    seed: int,
    device,
    dtype,
    enable_nvtx: bool,
) -> dict[str, object]:
    import torch

    generator = torch.Generator(device=device).manual_seed(seed + context_len + block_size)
    query = torch.randn(
        (batch_size, query_heads, 1, head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    contiguous_key = torch.randn(
        (batch_size, kv_heads, context_len, head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    contiguous_value = torch.randn(
        (batch_size, kv_heads, context_len, head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    paged_key, paged_value, block_tables = _pack_contiguous(
        contiguous_key,
        contiguous_value,
        block_size,
    )
    expected, _ = attention.forward_prefill(
        query,
        contiguous_key,
        contiguous_value,
        causal=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    gather_times = []
    attention_times = []
    actual = None
    for _ in range(repeats):
        gather_start_ns = time.monotonic_ns()
        with nvtx_range(
            nvtx_label("phase6", "gather", block_size=block_size, context_len=context_len),
            enabled=enable_nvtx,
        ):
            gathered_key, gathered_value = attention.gather_kv(
                paged_key,
                paged_value,
                block_tables,
                seq_lens=[context_len] * batch_size,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        gather_times.append((time.monotonic_ns() - gather_start_ns) / 1_000_000)

        attention_start_ns = time.monotonic_ns()
        with nvtx_range(
            nvtx_label("phase6", "attention", block_size=block_size, context_len=context_len),
            enabled=enable_nvtx,
        ):
            actual, _ = attention.forward_prefill(
                query,
                gathered_key,
                gathered_value,
                causal=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        attention_times.append((time.monotonic_ns() - attention_start_ns) / 1_000_000)
    if actual is None:
        raise RuntimeError("benchmark produced no attention output")
    max_abs_diff = (actual.float() - expected.float()).abs().max().item()
    gather_temp_bytes = (
        batch_size
        * kv_heads
        * context_len
        * head_dim
        * paged_key.element_size()
        * 2
    )
    return {
        "batch_size": batch_size,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "context_len": context_len,
        "block_size": block_size,
        "repeats": repeats,
        "gather_time_ms": sum(gather_times) / repeats,
        "attention_time_ms": sum(attention_times) / repeats,
        "gather_temp_bytes": gather_temp_bytes,
        "max_abs_diff": max_abs_diff,
    }


def _pack_contiguous(contiguous_key, contiguous_value, block_size: int):
    import torch

    batch, heads, seq_len, head_dim = contiguous_key.shape
    block_tables: list[list[int]] = []
    key_blocks: list[Any] = []
    value_blocks: list[Any] = []
    for batch_index in range(batch):
        block_ids = []
        for start in range(0, seq_len, block_size):
            block_id = len(key_blocks)
            block_ids.append(block_id)
            key_block = torch.zeros(
                (heads, block_size, head_dim),
                dtype=contiguous_key.dtype,
                device=contiguous_key.device,
            )
            value_block = torch.zeros_like(key_block)
            end = min(seq_len, start + block_size)
            block_tokens = end - start
            key_block[:, :block_tokens] = contiguous_key[batch_index, :, start:end]
            value_block[:, :block_tokens] = contiguous_value[batch_index, :, start:end]
            key_blocks.append(key_block)
            value_blocks.append(value_block)
        block_tables.append(block_ids)
    return torch.stack(key_blocks), torch.stack(value_blocks), block_tables


def _float_metric(case: dict[str, object], name: str) -> float:
    value = case[name]
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _int_metric(case: dict[str, object], name: str) -> int:
    value = case[name]
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_config(config: Phase6PagedAttentionBenchmarkConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.query_heads <= 0:
        raise ValueError("query_heads must be positive")
    if config.kv_heads <= 0:
        raise ValueError("kv_heads must be positive")
    if config.query_heads % config.kv_heads != 0:
        raise ValueError("query_heads must be a multiple of kv_heads")
    if config.head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if not config.context_lens or any(length <= 0 for length in config.context_lens):
        raise ValueError("context_lens must contain positive values")
    if not config.block_sizes or any(size <= 0 for size in config.block_sizes):
        raise ValueError("block_sizes must contain positive values")
    if config.repeats <= 0:
        raise ValueError("repeats must be positive")


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
