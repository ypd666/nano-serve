"""Phase 1 offline benchmark runner."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nano_serve.assets import AssetConfig
from nano_serve.benchmark.datasets import load_sharegpt_dataset
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import EngineConfig
from nano_serve.engine.core import Engine, PhaseEvent, StreamEvent
from nano_serve.model.tokenizer import TokenizerWrapper
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform
from nano_serve.sampling.base import SamplingParams


@dataclass(frozen=True)
class OfflineBenchmarkConfig:
    output_dir: Path
    num_samples: int = 1
    max_new_tokens: int = 8
    max_prompt_tokens: int = 128
    workload: str = "single_short"


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
        }


def run_offline_benchmark(config: OfflineBenchmarkConfig) -> dict[str, object]:
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
    )
    run_config = {
        "run_id": run_id,
        "phase": "phase1",
        "workload": config.workload,
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

        for sample in dataset.samples:
            prompt_token_ids = tokenizer.encode(sample.prompt)[: config.max_prompt_tokens]
            params = SamplingParams(
                max_tokens=config.max_new_tokens,
                stop_token_ids=_stop_token_ids(tokenizer),
            )
            before_finished = len(engine.finished)
            request_start_ns = time.monotonic_ns()
            stream_events: list[StreamEvent] = []

            def phase_callback(event: PhaseEvent) -> None:
                if event.phase == "prefill":
                    writer.write(
                        Event(
                            f"prefill_{event.event}",
                            fields={
                                "request_id": event.request_id,
                                "sample_id": sample.sample_id,
                                "num_tokens": event.num_tokens,
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
    elapsed_ns: int,
    platform: dict[str, object],
    artifacts: dict[str, str],
) -> dict[str, object]:
    total_input_tokens = sum(item.input_tokens for item in request_summaries)
    total_output_tokens = sum(item.output_tokens for item in request_summaries)
    elapsed_s = elapsed_ns / 1_000_000_000
    return {
        "run_id": run_id,
        "phase": "phase1",
        "workload": config.workload,
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
        "platform": platform,
        "artifacts": artifacts,
    }


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

