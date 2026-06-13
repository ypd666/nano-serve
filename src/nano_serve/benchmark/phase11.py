"""Phase 11 speculative decoding benchmark."""

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
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform
from nano_serve.speculative import (
    GreedyTokenStreamVerifier,
    SpeculativeDecodeConfig,
    SpeculativeDecoder,
    StaticDraftModel,
    decode_batch,
)


@dataclass(frozen=True)
class Phase11SpeculativeBenchmarkConfig:
    output_dir: Path
    gamma_values: tuple[int, ...] = (1, 2, 4, 8)
    output_tokens: int = 256
    batch_size: int = 4
    prompt_tokens: int = 16
    target_step_time_ms: float = 1.0
    draft_token_time_ms: float = 0.1


def run_phase11_speculative_benchmark(
    config: Phase11SpeculativeBenchmarkConfig,
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
    engine_config = EngineConfig(spec_decode="draft_model")
    run_config = {
        "run_id": run_id,
        "phase": "phase11",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "gamma_values": list(config.gamma_values),
        "output_tokens": config.output_tokens,
        "batch_size": config.batch_size,
        "prompt_tokens": config.prompt_tokens,
        "target_step_time_ms": config.target_step_time_ms,
        "draft_token_time_ms": config.draft_token_time_ms,
        "engine_config": engine_config.to_dict(),
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    cases: list[dict[str, object]] = []
    start_ns = time.monotonic_ns()
    with JSONLEventWriter(events_path) as writer:
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase11"}))
        writer.write(platform_event(platform_info))
        for workload in ("friendly", "hostile"):
            for gamma in config.gamma_values:
                case = _run_case(config, workload=workload, gamma=gamma, writer=writer)
                cases.append(case)
                writer.write(Event("speculative_case", fields=case))
        end_ns = time.monotonic_ns()
        summary = {
            "run_id": run_id,
            "phase": "phase11",
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": (end_ns - start_ns) / 1_000_000_000,
            "workload": "speculative_friendly_hostile",
            "scheduler": "continuous",
            "kv_cache": "paged_prefix",
            "cases": cases,
            "best_speedup_case": max(
                cases,
                key=lambda item: _float_metric(item, "estimated_speedup"),
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
    config: Phase11SpeculativeBenchmarkConfig,
    *,
    workload: str,
    gamma: int,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    target_streams = [_target_stream(config, request_index) for request_index in range(config.batch_size)]
    draft_streams = [
        _draft_stream(target_stream, workload=workload, request_index=request_index)
        for request_index, target_stream in enumerate(target_streams)
    ]
    prompt_batch = [
        [100_000 + request_index * config.prompt_tokens + token_index for token_index in range(config.prompt_tokens)]
        for request_index in range(config.batch_size)
    ]
    decoders = [
        SpeculativeDecoder(
            StaticDraftModel(draft_stream, base_context_len=config.prompt_tokens),
            GreedyTokenStreamVerifier(target_stream, base_context_len=config.prompt_tokens),
        )
        for draft_stream, target_stream in zip(draft_streams, target_streams, strict=True)
    ]
    results = decode_batch(
        decoders,
        prompt_batch,
        config=SpeculativeDecodeConfig(gamma=gamma, max_tokens=config.output_tokens),
    )
    total_target_calls = 0
    total_draft_tokens = 0
    total_output_tokens = 0
    total_accepted = 0
    total_rejections = 0
    total_bonus_tokens = 0
    total_rollback_tokens = 0
    total_kv_appended = 0
    acceptance_lengths: list[int] = []
    for request_index, result in enumerate(results):
        metrics = result.metrics
        total_target_calls += metrics.target_calls
        total_draft_tokens += metrics.draft_tokens_proposed
        total_output_tokens += len(result.output_token_ids)
        total_accepted += metrics.accepted_tokens
        total_rejections += metrics.rejection_count
        total_bonus_tokens += metrics.bonus_tokens
        total_rollback_tokens += metrics.rollback_tokens
        total_kv_appended += metrics.kv_tokens_appended
        acceptance_lengths.extend(metrics.acceptance_lengths)
        writer.write(
            Event(
                "speculative_request_end",
                fields={
                    "workload": workload,
                    "gamma": gamma,
                    "request_index": request_index,
                    "output_tokens": len(result.output_token_ids),
                    **metrics.to_dict(),
                },
            )
        )
        for iteration_index, verification in enumerate(result.iterations):
            writer.write(
                Event(
                    "speculative_iteration",
                    fields={
                        "workload": workload,
                        "gamma": gamma,
                        "request_index": request_index,
                        "iteration": iteration_index,
                        **verification.to_dict(),
                    },
                )
            )

    baseline_time_ms = total_output_tokens * config.target_step_time_ms
    speculative_time_ms = (
        total_target_calls * config.target_step_time_ms
        + total_draft_tokens * config.draft_token_time_ms
    )
    return {
        "workload": workload,
        "gamma": gamma,
        "batch_size": config.batch_size,
        "output_tokens": total_output_tokens,
        "draft_tokens_proposed": total_draft_tokens,
        "accepted_tokens": total_accepted,
        "acceptance_rate": total_accepted / total_draft_tokens if total_draft_tokens else 0.0,
        "mean_acceptance_length": sum(acceptance_lengths) / len(acceptance_lengths)
        if acceptance_lengths
        else 0.0,
        "target_calls": total_target_calls,
        "target_calls_per_output_token": total_target_calls / total_output_tokens
        if total_output_tokens
        else 0.0,
        "rejection_count": total_rejections,
        "bonus_tokens": total_bonus_tokens,
        "rollback_tokens": total_rollback_tokens,
        "kv_tokens_appended": total_kv_appended,
        "baseline_time_ms": baseline_time_ms,
        "speculative_time_ms": speculative_time_ms,
        "estimated_speedup": baseline_time_ms / speculative_time_ms
        if speculative_time_ms
        else 0.0,
    }


def _target_stream(config: Phase11SpeculativeBenchmarkConfig, request_index: int) -> list[int]:
    return [
        request_index * 10_000 + token_index
        for token_index in range(config.output_tokens + max(config.gamma_values) + 1)
    ]


def _draft_stream(target_stream: list[int], *, workload: str, request_index: int) -> list[int]:
    if workload == "friendly":
        return [
            token if (index + request_index) % 16 else token + 1
            for index, token in enumerate(target_stream)
        ]
    if workload == "hostile":
        return [token + 1 for token in target_stream]
    raise ValueError(f"unknown workload: {workload}")


def _validate_config(config: Phase11SpeculativeBenchmarkConfig) -> None:
    if not config.gamma_values or any(value <= 0 for value in config.gamma_values):
        raise ValueError("gamma_values must contain positive values")
    if config.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.prompt_tokens < 0:
        raise ValueError("prompt_tokens must be non-negative")
    if config.target_step_time_ms <= 0:
        raise ValueError("target_step_time_ms must be positive")
    if config.draft_token_time_ms < 0:
        raise ValueError("draft_token_time_ms must be non-negative")


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
