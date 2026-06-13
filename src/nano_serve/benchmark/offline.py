"""Phase 1 offline benchmark runner."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nano_serve.assets import AssetConfig
from nano_serve.benchmark.datasets import ServingSample, load_sharegpt_dataset
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import EngineConfig, KVCacheKind, SchedulerKind
from nano_serve.engine.core import BatchEvent, Engine, PhaseEvent, StreamEvent
from nano_serve.model.tokenizer import TokenizerWrapper
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform
from nano_serve.sampling.base import SamplingParams
from nano_serve.scheduler.policies import SchedulerPolicy


@dataclass(frozen=True)
class OfflineBenchmarkConfig:
    output_dir: Path
    num_samples: int = 1
    max_new_tokens: int = 8
    max_prompt_tokens: int = 128
    workload: str = "single_short"
    kv_cache: KVCacheKind = "none"
    scheduler: SchedulerKind = "single"
    scheduler_policy: SchedulerPolicy = SchedulerPolicy.FCFS
    batch_size: int = 1
    max_num_batched_tokens: int = 4096


@dataclass(frozen=True)
class RequestBenchmarkSummary:
    sample_id: str
    source_index: int
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float | None
    wall_ms: float
    kv_cache: str
    kv_sequence_length: int | None = None
    kv_bytes_used: int | None = None
    kv_blocks_used: int | None = None
    kv_fragmentation: float | None = None
    batch_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "source_index": self.source_index,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stop_reason": self.stop_reason,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
            "wall_ms": self.wall_ms,
            "kv_cache": self.kv_cache,
            "kv_sequence_length": self.kv_sequence_length,
            "kv_bytes_used": self.kv_bytes_used,
            "kv_blocks_used": self.kv_blocks_used,
            "kv_fragmentation": self.kv_fragmentation,
            "batch_id": self.batch_id,
        }


@dataclass(frozen=True)
class BatchBenchmarkSummary:
    batch_id: int
    batch_size: int
    model_invocations: int
    decode_invocations: int
    total_real_tokens: int
    total_padded_tokens: int
    total_inactive_slot_steps: int
    max_tokens_per_slot: int
    total_cpu_schedule_time_ms: float = 0.0
    max_running_reqs: int = 0
    max_waiting_reqs: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "model_invocations": self.model_invocations,
            "decode_invocations": self.decode_invocations,
            "total_real_tokens": self.total_real_tokens,
            "total_padded_tokens": self.total_padded_tokens,
            "total_inactive_slot_steps": self.total_inactive_slot_steps,
            "max_tokens_per_slot": self.max_tokens_per_slot,
            "total_cpu_schedule_time_ms": self.total_cpu_schedule_time_ms,
            "max_running_reqs": self.max_running_reqs,
            "max_waiting_reqs": self.max_waiting_reqs,
        }


def run_offline_benchmark(config: OfflineBenchmarkConfig) -> dict[str, object]:
    _validate_offline_config(config)
    asset_config = AssetConfig.from_env()
    platform_info = detect_platform()
    run_id = _run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"

    engine_config = EngineConfig(
        model_path=str(asset_config.model_path),
        dataset_path=str(asset_config.dataset_path),
        kv_cache=config.kv_cache,
        scheduler=config.scheduler,
        scheduler_policy=config.scheduler_policy,
        max_num_seqs=config.batch_size,
        max_num_batched_tokens=config.max_num_batched_tokens,
    )
    run_config = {
        "run_id": run_id,
        "phase": _phase_name(config),
        "workload": config.workload,
        "kv_cache": config.kv_cache,
        "scheduler": config.scheduler,
        "scheduler_policy": config.scheduler_policy.value,
        "batch_size": config.batch_size,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "num_samples": config.num_samples,
        "max_new_tokens": config.max_new_tokens,
        "max_prompt_tokens": config.max_prompt_tokens,
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "engine_config": engine_config.to_dict(),
        "asset_config": {
            "model_path": str(asset_config.model_path),
            "dataset_path": str(asset_config.dataset_path),
            "model_id": asset_config.model_id,
            "dataset_repo_id": asset_config.dataset_repo_id,
            "dataset_filename": asset_config.dataset_filename,
        },
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    tokenizer = TokenizerWrapper.from_pretrained(asset_config.model_path)
    dataset = load_sharegpt_dataset(asset_config.dataset_path, max_samples=config.num_samples)
    engine = Engine(engine_config)

    request_summaries: list[RequestBenchmarkSummary] = []
    batch_summaries: list[BatchBenchmarkSummary] = []
    run_start_ns = time.monotonic_ns()
    with JSONLEventWriter(events_path) as writer:
        writer.write(Event("run_start", fields={"run_id": run_id, "workload": config.workload}))
        writer.write(platform_event(platform_info))
        writer.write(
            Event(
                "dataset_load_end",
                fields={
                    "path": dataset.path,
                    "samples_loaded": len(dataset.samples),
                    "samples_skipped": dataset.skipped,
                    "dataset_total": dataset.total,
                },
            )
        )

        if config.scheduler == "continuous":
            continuous_request_summaries, continuous_batch_summary = _run_continuous_batch(
                samples=dataset.samples,
                batch_id=0,
                config=config,
                tokenizer=tokenizer,
                engine=engine,
                writer=writer,
            )
            request_summaries.extend(continuous_request_summaries)
            batch_summaries.append(continuous_batch_summary)

        for batch_id, samples in enumerate(_sample_batches(dataset.samples, config.batch_size)):
            if config.scheduler == "continuous":
                break
            if config.scheduler == "static_batch":
                batch_request_summaries, batch_summary = _run_static_batch(
                    samples=samples,
                    batch_id=batch_id,
                    config=config,
                    tokenizer=tokenizer,
                    engine=engine,
                    writer=writer,
                )
                request_summaries.extend(batch_request_summaries)
                batch_summaries.append(batch_summary)
                continue

            sample = samples[0]
            prompt_token_ids = tokenizer.encode(sample.prompt)[: config.max_prompt_tokens]
            params = SamplingParams(
                max_tokens=config.max_new_tokens,
                stop_token_ids=_stop_token_ids(tokenizer),
            )
            before_finished = len(engine.finished)
            request_start_ns = time.monotonic_ns()
            stream_events: list[StreamEvent] = []

            def phase_callback(event: PhaseEvent) -> None:
                metadata = event.metadata or {}
                if event.phase == "prefill":
                    writer.write(
                        Event(
                            f"prefill_{event.event}",
                            fields={
                                "request_id": event.request_id,
                                "sample_id": sample.sample_id,
                                "num_tokens": event.num_tokens,
                                **metadata,
                            },
                        )
                    )
                    return

                if event.phase == "decode":
                    writer.write(
                        Event(
                            f"decode_step_{event.event}",
                            fields={
                                "request_id": event.request_id,
                                "sample_id": sample.sample_id,
                                "token_index": event.token_index,
                                "num_tokens": event.num_tokens,
                                **metadata,
                            },
                        )
                    )

            def stream_callback(event: StreamEvent) -> None:
                stream_events.append(event)
                writer.write(
                    Event(
                        "stream_token",
                        fields={
                            "request_id": event.request_id,
                            "sample_id": sample.sample_id,
                            "token_id": event.token_id,
                            "token_index": event.token_index,
                        },
                    )
                )

            output_token_ids = engine.generate(
                prompt_token_ids,
                params,
                stream_callback,
                phase_callback,
            )
            request_end_ns = time.monotonic_ns()
            state = engine.finished[before_finished]
            kv_metadata = _latest_kv_metadata(state)
            if len(stream_events) != len(output_token_ids):
                raise RuntimeError("stream callback count does not match generated tokens")
            request_summary = RequestBenchmarkSummary(
                sample_id=sample.sample_id,
                source_index=sample.source_index,
                input_tokens=len(prompt_token_ids),
                output_tokens=len(output_token_ids),
                stop_reason=state.stop_reason,
                ttft_ms=state.metrics.ttft_ms,
                tpot_ms=state.metrics.tpot_ms(len(output_token_ids)),
                e2e_ms=state.metrics.e2e_ms,
                wall_ms=(request_end_ns - request_start_ns) / 1_000_000,
                kv_cache=config.kv_cache,
                kv_sequence_length=_optional_int(kv_metadata.get("kv_sequence_length")),
                kv_bytes_used=_optional_int(kv_metadata.get("kv_bytes_used")),
                kv_blocks_used=_optional_int(kv_metadata.get("kv_blocks_used")),
                kv_fragmentation=_optional_float(kv_metadata.get("kv_fragmentation")),
                batch_id=batch_id,
            )
            request_summaries.append(request_summary)
            writer.write(Event("request_end", fields=request_summary.to_dict()))

        run_end_ns = time.monotonic_ns()
        summary = _summary(
            run_id=run_id,
            run_dir=run_dir,
            config=config,
            dataset_total=dataset.total,
            request_summaries=request_summaries,
            batch_summaries=batch_summaries,
            elapsed_ns=run_end_ns - run_start_ns,
            platform=platform_info.to_dict(),
            artifacts={
                "run_config": str(run_config_path),
                "events": str(events_path),
                "summary": str(summary_path),
                "report": str(report_path),
            },
        )
        writer.write(Event("run_end", fields=summary))

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report_path, summary)
    return summary


def _summary(
    *,
    run_id: str,
    run_dir: Path,
    config: OfflineBenchmarkConfig,
    dataset_total: int,
    request_summaries: list[RequestBenchmarkSummary],
    batch_summaries: list[BatchBenchmarkSummary],
    elapsed_ns: int,
    platform: dict[str, object],
    artifacts: dict[str, str],
) -> dict[str, object]:
    total_input_tokens = sum(item.input_tokens for item in request_summaries)
    total_output_tokens = sum(item.output_tokens for item in request_summaries)
    elapsed_s = elapsed_ns / 1_000_000_000
    return {
        "run_id": run_id,
        "phase": _phase_name(config),
        "workload": config.workload,
        "kv_cache": config.kv_cache,
        "scheduler": config.scheduler,
        "scheduler_policy": config.scheduler_policy.value,
        "batch_size": config.batch_size,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "status": "ok",
        "run_dir": str(run_dir),
        "samples_loaded": len(request_summaries),
        "dataset_total": dataset_total,
        "max_new_tokens": config.max_new_tokens,
        "max_prompt_tokens": config.max_prompt_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_s": elapsed_s,
        "requests_per_sec": len(request_summaries) / elapsed_s if elapsed_s else None,
        "output_tokens_per_sec": total_output_tokens / elapsed_s if elapsed_s else None,
        "total_tokens_per_sec": (total_input_tokens + total_output_tokens) / elapsed_s
        if elapsed_s
        else None,
        "requests": [item.to_dict() for item in request_summaries],
        "batches": [item.to_dict() for item in batch_summaries],
        "batch_count": len(batch_summaries),
        "max_batch_size": max(
            (item.batch_size for item in batch_summaries),
            default=1 if request_summaries else 0,
        ),
        "total_padded_tokens": sum(item.total_padded_tokens for item in batch_summaries),
        "total_inactive_slot_steps": sum(
            item.total_inactive_slot_steps for item in batch_summaries
        ),
        "total_cpu_schedule_time_ms": sum(
            item.total_cpu_schedule_time_ms for item in batch_summaries
        ),
        "max_running_reqs": max(
            (item.max_running_reqs for item in batch_summaries),
            default=0,
        ),
        "max_waiting_reqs": max(
            (item.max_waiting_reqs for item in batch_summaries),
            default=0,
        ),
        "max_kv_bytes_used": max(
            (item.kv_bytes_used or 0 for item in request_summaries),
            default=0,
        ),
        "platform": platform,
        "artifacts": artifacts,
    }


def _run_static_batch(
    *,
    samples: list[ServingSample],
    batch_id: int,
    config: OfflineBenchmarkConfig,
    tokenizer: TokenizerWrapper,
    engine: Engine,
    writer: JSONLEventWriter,
) -> tuple[list[RequestBenchmarkSummary], BatchBenchmarkSummary]:
    prompt_token_ids_batch = [
        tokenizer.encode(sample.prompt)[: config.max_prompt_tokens] for sample in samples
    ]
    sampling_params = [
        SamplingParams(
            max_tokens=config.max_new_tokens,
            stop_token_ids=_stop_token_ids(tokenizer),
        )
        for _ in samples
    ]
    request_ids = [
        f"batch-{batch_id}-slot-{slot}-sample-{sample.sample_id}"
        for slot, sample in enumerate(samples)
    ]
    request_lookup = {
        request_id: (slot, sample)
        for slot, (request_id, sample) in enumerate(zip(request_ids, samples, strict=True))
    }
    stream_events: dict[str, list[StreamEvent]] = {request_id: [] for request_id in request_ids}
    batch_events: list[BatchEvent] = []
    before_finished = len(engine.finished)
    batch_start_ns = time.monotonic_ns()

    def stream_callback(event: StreamEvent) -> None:
        stream_events[event.request_id].append(event)
        slot, sample = request_lookup[event.request_id]
        writer.write(
            Event(
                "stream_token",
                fields={
                    "request_id": event.request_id,
                    "sample_id": sample.sample_id,
                    "batch_id": batch_id,
                    "slot": slot,
                    "token_id": event.token_id,
                    "token_index": event.token_index,
                },
            )
        )

    def batch_callback(event: BatchEvent) -> None:
        batch_events.append(event)
        event_name = event.event if event.event.startswith("batch_") else f"batch_{event.event}"
        writer.write(
            Event(
                event_name,
                fields={
                    "batch_id": batch_id,
                    "iteration": event.iteration,
                    "engine_timestamp_ns": event.timestamp_ns,
                    **event.metadata,
                },
            )
        )

    output_token_ids_batch = engine.generate_static_batch(
        list(zip(prompt_token_ids_batch, sampling_params, strict=True)),
        request_ids=request_ids,
        stream_callback=stream_callback,
        batch_callback=batch_callback,
    )
    batch_end_ns = time.monotonic_ns()
    states = engine.finished[before_finished : before_finished + len(samples)]
    if len(states) != len(samples):
        raise RuntimeError("static batch finished state count mismatch")

    request_summaries: list[RequestBenchmarkSummary] = []
    for slot, (sample, prompt_token_ids, output_token_ids, state) in enumerate(
        zip(samples, prompt_token_ids_batch, output_token_ids_batch, states, strict=True)
    ):
        request_id = request_ids[slot]
        if state.request_id != request_id:
            raise RuntimeError("static batch finished state order mismatch")
        if len(stream_events[request_id]) != len(output_token_ids):
            raise RuntimeError("stream callback count does not match generated tokens")

        request_summary = RequestBenchmarkSummary(
            sample_id=sample.sample_id,
            source_index=sample.source_index,
            input_tokens=len(prompt_token_ids),
            output_tokens=len(output_token_ids),
            stop_reason=state.stop_reason,
            ttft_ms=state.metrics.ttft_ms,
            tpot_ms=state.metrics.tpot_ms(len(output_token_ids)),
            e2e_ms=state.metrics.e2e_ms,
            wall_ms=(batch_end_ns - batch_start_ns) / 1_000_000,
            kv_cache=config.kv_cache,
            batch_id=batch_id,
        )
        request_summaries.append(request_summary)
        writer.write(
            Event(
                "batch_request_end",
                fields={
                    **request_summary.to_dict(),
                    "request_id": request_id,
                    "slot": slot,
                },
            )
        )

    return request_summaries, _batch_summary_from_events(batch_id, batch_events)


def _run_continuous_batch(
    *,
    samples: list[ServingSample],
    batch_id: int,
    config: OfflineBenchmarkConfig,
    tokenizer: TokenizerWrapper,
    engine: Engine,
    writer: JSONLEventWriter,
) -> tuple[list[RequestBenchmarkSummary], BatchBenchmarkSummary]:
    prompt_token_ids_batch = [
        tokenizer.encode(sample.prompt)[: config.max_prompt_tokens] for sample in samples
    ]
    sampling_params = [
        SamplingParams(
            max_tokens=config.max_new_tokens,
            stop_token_ids=_stop_token_ids(tokenizer),
        )
        for _ in samples
    ]
    request_ids = [
        f"continuous-{batch_id}-req-{slot}-sample-{sample.sample_id}"
        for slot, sample in enumerate(samples)
    ]
    request_lookup = {
        request_id: (slot, sample)
        for slot, (request_id, sample) in enumerate(zip(request_ids, samples, strict=True))
    }
    stream_events: dict[str, list[StreamEvent]] = {request_id: [] for request_id in request_ids}
    batch_events: list[BatchEvent] = []
    before_finished = len(engine.finished)
    batch_start_ns = time.monotonic_ns()

    def stream_callback(event: StreamEvent) -> None:
        stream_events[event.request_id].append(event)
        slot, sample = request_lookup[event.request_id]
        writer.write(
            Event(
                "stream_token",
                fields={
                    "request_id": event.request_id,
                    "sample_id": sample.sample_id,
                    "batch_id": batch_id,
                    "slot": slot,
                    "token_id": event.token_id,
                    "token_index": event.token_index,
                },
            )
        )

    def batch_callback(event: BatchEvent) -> None:
        batch_events.append(event)
        writer.write(
            Event(
                f"continuous_{event.event}",
                fields={
                    "batch_id": batch_id,
                    "iteration": event.iteration,
                    "engine_timestamp_ns": event.timestamp_ns,
                    **event.metadata,
                },
            )
        )

    output_token_ids_batch = engine.generate_continuous(
        list(zip(prompt_token_ids_batch, sampling_params, strict=True)),
        request_ids=request_ids,
        stream_callback=stream_callback,
        batch_callback=batch_callback,
    )
    batch_end_ns = time.monotonic_ns()
    states = engine.finished[before_finished : before_finished + len(samples)]
    if len(states) != len(samples):
        raise RuntimeError("continuous finished state count mismatch")
    states_by_id = {state.request_id: state for state in states}

    request_summaries: list[RequestBenchmarkSummary] = []
    for slot, (sample, prompt_token_ids, output_token_ids, request_id) in enumerate(
        zip(samples, prompt_token_ids_batch, output_token_ids_batch, request_ids, strict=True)
    ):
        state = states_by_id[request_id]
        if len(stream_events[request_id]) != len(output_token_ids):
            raise RuntimeError("stream callback count does not match generated tokens")

        request_summary = RequestBenchmarkSummary(
            sample_id=sample.sample_id,
            source_index=sample.source_index,
            input_tokens=len(prompt_token_ids),
            output_tokens=len(output_token_ids),
            stop_reason=state.stop_reason,
            ttft_ms=state.metrics.ttft_ms,
            tpot_ms=state.metrics.tpot_ms(len(output_token_ids)),
            e2e_ms=state.metrics.e2e_ms,
            wall_ms=(batch_end_ns - batch_start_ns) / 1_000_000,
            kv_cache=config.kv_cache,
            batch_id=batch_id,
        )
        request_summaries.append(request_summary)
        writer.write(
            Event(
                "continuous_request_end",
                fields={
                    **request_summary.to_dict(),
                    "request_id": request_id,
                    "slot": slot,
                },
            )
        )

    return request_summaries, _continuous_batch_summary_from_events(batch_id, batch_events)


def _batch_summary_from_events(
    batch_id: int,
    batch_events: list[BatchEvent],
) -> BatchBenchmarkSummary:
    model_invocations = [
        event
        for event in batch_events
        if event.event in {"prefill_start", "decode_step_start"}
    ]
    return BatchBenchmarkSummary(
        batch_id=batch_id,
        batch_size=max(
            (_optional_int(event.metadata.get("batch_size")) or 0 for event in batch_events),
            default=0,
        ),
        model_invocations=len(model_invocations),
        decode_invocations=sum(
            1 for event in model_invocations if event.event == "decode_step_start"
        ),
        total_real_tokens=sum(
            _optional_int(event.metadata.get("real_tokens")) or 0
            for event in model_invocations
        ),
        total_padded_tokens=sum(
            _optional_int(event.metadata.get("padded_tokens")) or 0
            for event in model_invocations
        ),
        total_inactive_slot_steps=sum(
            _optional_int(event.metadata.get("inactive_slots")) or 0
            for event in model_invocations
        ),
        max_tokens_per_slot=max(
            (
                _optional_int(event.metadata.get("max_tokens_per_slot")) or 0
                for event in batch_events
            ),
            default=0,
        ),
    )


def _continuous_batch_summary_from_events(
    batch_id: int,
    batch_events: list[BatchEvent],
) -> BatchBenchmarkSummary:
    starts = [event for event in batch_events if event.event == "iteration_start"]
    return BatchBenchmarkSummary(
        batch_id=batch_id,
        batch_size=max(
            (_optional_int(event.metadata.get("batch_size")) or 0 for event in starts),
            default=0,
        ),
        model_invocations=len(starts),
        decode_invocations=sum(
            1 for event in starts if _optional_int(event.metadata.get("num_decode_tokens"))
        ),
        total_real_tokens=sum(
            _optional_int(event.metadata.get("real_tokens")) or 0 for event in starts
        ),
        total_padded_tokens=sum(
            _optional_int(event.metadata.get("padded_tokens")) or 0 for event in starts
        ),
        total_inactive_slot_steps=0,
        max_tokens_per_slot=max(
            (_optional_int(event.metadata.get("max_tokens_per_slot")) or 0 for event in starts),
            default=0,
        ),
        total_cpu_schedule_time_ms=sum(
            _optional_float(event.metadata.get("cpu_schedule_time_ms")) or 0.0
            for event in starts
        ),
        max_running_reqs=max(
            (_optional_int(event.metadata.get("num_running_reqs")) or 0 for event in starts),
            default=0,
        ),
        max_waiting_reqs=max(
            (_optional_int(event.metadata.get("num_waiting_reqs")) or 0 for event in starts),
            default=0,
        ),
    )


def _sample_batches(
    samples: list[ServingSample],
    batch_size: int,
) -> Iterator[list[ServingSample]]:
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def _validate_offline_config(config: OfflineBenchmarkConfig) -> None:
    if config.num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if config.max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.scheduler == "static_batch" and config.kv_cache != "none":
        raise ValueError("Phase 3 static batching currently requires kv_cache='none'.")
    if config.scheduler == "continuous" and config.kv_cache != "none":
        raise ValueError("Phase 4 continuous batching currently requires kv_cache='none'.")
    if config.max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")


def _effective_batch_size(config: OfflineBenchmarkConfig) -> int:
    return config.batch_size if config.scheduler in {"static_batch", "continuous"} else 1


def _phase_name(config: OfflineBenchmarkConfig) -> str:
    if config.scheduler == "continuous":
        return "phase4"
    if config.scheduler == "static_batch":
        return "phase3"
    return "phase1"


def _latest_kv_metadata(state: object) -> dict[str, object]:
    events = getattr(state, "phase_metadata", None)
    if not isinstance(events, list) or not events:
        return {}
    latest = events[-1]
    return latest if isinstance(latest, dict) else {}


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _stop_token_ids(tokenizer: TokenizerWrapper) -> tuple[int, ...]:
    if tokenizer.eos_token_id is None:
        return ()
    return (tokenizer.eos_token_id,)


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

