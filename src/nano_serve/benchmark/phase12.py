"""Phase 12 advanced serving feature benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nano_serve.advanced import (
    JSONGrammarState,
    KVQuantizer,
    LoRAAdapter,
    LoRAAdapterRegistry,
    StructuredLogitsProcessor,
    WeightQuantizer,
)
from nano_serve.benchmark.profiler import nvtx_label, nvtx_range
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import AdvancedFeatureConfig, BenchmarkConfig, EngineConfig
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform


@dataclass(frozen=True)
class Phase12AdvancedBenchmarkConfig:
    output_dir: Path
    hidden_size: int = 512
    rank: int = 8
    tokens: int = 1024
    batch_size: int = 8
    seed: int = 0
    enable_nvtx: bool = False


def run_phase12_advanced_benchmark(
    config: Phase12AdvancedBenchmarkConfig,
) -> dict[str, object]:
    _validate_config(config)
    import torch

    torch.manual_seed(config.seed)
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
        advanced=AdvancedFeatureConfig(
            weight_quantization="int8",
            kv_quantization="int8",
            lora=True,
            structured_output="json_object",
        ),
        benchmark=BenchmarkConfig(enable_nvtx=config.enable_nvtx),
    )
    run_config = {
        "run_id": run_id,
        "phase": "phase12",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "hidden_size": config.hidden_size,
        "rank": config.rank,
        "tokens": config.tokens,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "enable_nvtx": config.enable_nvtx,
        "engine_config": engine_config.to_dict(),
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    start_ns = time.monotonic_ns()
    with (
        JSONLEventWriter(events_path) as writer,
        nvtx_range(nvtx_label("phase12", "run"), enabled=config.enable_nvtx),
    ):
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase12"}))
        writer.write(platform_event(platform_info))
        with nvtx_range(
            nvtx_label("phase12", "case", case="quantization"),
            enabled=config.enable_nvtx,
        ):
            quant_cases = _run_quant_cases(config, writer)
        with nvtx_range(
            nvtx_label("phase12", "case", case="lora"),
            enabled=config.enable_nvtx,
        ):
            lora_case = _run_lora_case(config, writer)
        with nvtx_range(
            nvtx_label("phase12", "case", case="structured_output"),
            enabled=config.enable_nvtx,
        ):
            structured_case = _run_structured_case(config, writer)
        end_ns = time.monotonic_ns()
        summary = {
            "run_id": run_id,
            "phase": "phase12",
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": (end_ns - start_ns) / 1_000_000_000,
            "workload": "advanced_serving_reference",
            "scheduler": "continuous",
            "kv_cache": "paged_prefix",
            "quant_cases": quant_cases,
            "lora_case": lora_case,
            "structured_case": structured_case,
            "best_memory_saving_case": max(
                quant_cases,
                key=lambda item: _float_metric(item, "memory_saving_ratio"),
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


def _run_quant_cases(
    config: Phase12AdvancedBenchmarkConfig,
    writer: JSONLEventWriter,
) -> list[dict[str, object]]:
    import torch

    weight = torch.randn(config.hidden_size, config.hidden_size)
    kv = torch.randn(config.tokens, 2, 64)
    cases: list[dict[str, object]] = []
    for dtype in ("int8", "int4"):
        quantizer = WeightQuantizer(dtype=dtype, axis=1)
        with nvtx_range(
            nvtx_label("phase12", "quant_weight", dtype=dtype),
            enabled=config.enable_nvtx,
        ):
            start_ns = time.monotonic_ns()
            quantized = quantizer.quantize(weight)
            restored = quantizer.dequantize(quantized)
            end_ns = time.monotonic_ns()
        diff = (restored - weight).abs()
        case = {
            "case": f"weight_only_{dtype}",
            "quantization": quantized.to_dict(),
            "memory_saving_ratio": quantized.memory_saving_ratio,
            "max_abs_error": float(diff.max().item()),
            "mean_abs_error": float(diff.mean().item()),
            "elapsed_ms": (end_ns - start_ns) / 1_000_000,
            "baseline": "float32_weight",
        }
        cases.append(case)
        writer.write(Event("phase12_quant_case", fields=case))

    kv_dtypes: tuple[Literal["int8", "fp8"], ...] = ("int8", "fp8")
    for dtype in kv_dtypes:
        kv_quantizer = KVQuantizer(dtype=dtype)
        with nvtx_range(
            nvtx_label("phase12", "quant_kv", dtype=dtype),
            enabled=config.enable_nvtx,
        ):
            start_ns = time.monotonic_ns()
            quantized_kv = kv_quantizer.quantize(kv)
            restored_kv = kv_quantizer.dequantize(quantized_kv)
            end_ns = time.monotonic_ns()
        kv_diff = (restored_kv - kv).abs()
        kv_case = {
            "case": f"kv_{dtype}",
            "quantization": quantized_kv.to_dict(),
            "memory_saving_ratio": quantized_kv.memory_saving_ratio,
            "max_abs_error": float(kv_diff.max().item()),
            "mean_abs_error": float(kv_diff.mean().item()),
            "elapsed_ms": (end_ns - start_ns) / 1_000_000,
            "baseline": "float32_kv",
        }
        cases.append(kv_case)
        writer.write(Event("phase12_quant_case", fields=kv_case))
    return cases


def _run_lora_case(
    config: Phase12AdvancedBenchmarkConfig,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    import torch

    registry = LoRAAdapterRegistry()
    for adapter_index in range(2):
        registry.register(
            LoRAAdapter(
                adapter_id=f"adapter_{adapter_index}",
                a=torch.randn(config.hidden_size, config.rank) * 0.01,
                b=torch.randn(config.rank, config.hidden_size) * 0.01,
                alpha=float(config.rank),
            )
        )
    x = torch.randn(config.batch_size, config.hidden_size)
    adapter_ids = [
        f"adapter_{row % 2}"
        for row in range(config.batch_size)
    ]
    start_ns = time.monotonic_ns()
    batch_output = registry.apply_batch(x, adapter_ids)
    end_ns = time.monotonic_ns()
    manual = torch.cat(
        [
            x[row : row + 1] + registry.get(adapter_id).apply(x[row : row + 1])
            for row, adapter_id in enumerate(adapter_ids)
        ],
        dim=0,
    )
    isolation_diff = (batch_output - manual).abs()
    baseline_bytes = config.hidden_size * config.hidden_size * 4 * len(registry)
    adapter_bytes = (
        (config.hidden_size * config.rank + config.rank * config.hidden_size)
        * 4
        * len(registry)
    )
    case = {
        "case": "multi_lora_batch",
        "adapter_count": len(registry),
        "rank": config.rank,
        "batch_size": config.batch_size,
        "adapter_ids": adapter_ids,
        "switch_count": registry.switch_count(adapter_ids),
        "adapter_bytes": adapter_bytes,
        "dense_delta_bytes": baseline_bytes,
        "memory_saving_ratio": 1.0 - adapter_bytes / baseline_bytes,
        "max_abs_diff": float(isolation_diff.max().item()),
        "mean_abs_diff": float(isolation_diff.mean().item()),
        "elapsed_ms": (end_ns - start_ns) / 1_000_000,
    }
    writer.write(Event("phase12_lora_case", fields=case))
    return case


def _run_structured_case(
    config: Phase12AdvancedBenchmarkConfig,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    import torch

    processor = StructuredLogitsProcessor()
    state = JSONGrammarState()
    logits = torch.arange(0, 16, dtype=torch.float32)
    valid_sequence = [0, 4, 2, 5, 3, 4, 2, 7, 1]
    invalid_candidates = [5, 1, 4, 3, 2, 0]
    accepted = 0
    rejected = 0
    mask_time_ns = 0
    transition_trace: list[dict[str, Any]] = []
    for step, token_id in enumerate(valid_sequence):
        invalid_token = invalid_candidates[step % len(invalid_candidates)]
        if processor.accepts(state, invalid_token):
            accepted += 1
        else:
            rejected += 1
        start_ns = time.monotonic_ns()
        masked = processor.mask_logits(logits, state)
        mask_time_ns += time.monotonic_ns() - start_ns
        allowed = sorted(processor.allowed_token_ids(state))
        next_state = processor.advance(state, token_id)
        transition_trace.append(
            {
                "step": step,
                "state": state.state.value,
                "allowed_token_ids": allowed,
                "token_id": token_id,
                "invalid_probe_token_id": invalid_token,
                "masked_valid_tokens": int(torch.isfinite(masked).sum().item()),
                "next_state": next_state.state.value,
            }
        )
        if next_state.invalid:
            raise RuntimeError("valid structured output sequence was rejected")
        accepted += 1
        state = next_state
    case = {
        "case": "json_object_grammar",
        "tokens": len(valid_sequence),
        "accepted_tokens": accepted,
        "rejected_tokens": rejected,
        "final_state": state.state.value,
        "done": state.done,
        "mask_elapsed_ms": mask_time_ns / 1_000_000,
        "avg_mask_elapsed_ms": (mask_time_ns / 1_000_000) / len(valid_sequence),
        "transition_trace": transition_trace,
    }
    writer.write(Event("phase12_structured_case", fields=case))
    return case


def _validate_config(config: Phase12AdvancedBenchmarkConfig) -> None:
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.rank <= 0:
        raise ValueError("rank must be positive")
    if config.rank > config.hidden_size:
        raise ValueError("rank must be <= hidden_size")
    if config.tokens <= 0:
        raise ValueError("tokens must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _float_metric(case: dict[str, object], name: str) -> float:
    value = case[name]
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


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
