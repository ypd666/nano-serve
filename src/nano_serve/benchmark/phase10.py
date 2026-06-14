"""Phase 10 CPU/GPU overlap and graph benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nano_serve.benchmark.profiler import nvtx_label, nvtx_range
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import BenchmarkConfig, EngineConfig
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform
from nano_serve.runtime import (
    AsyncSchedulerPrep,
    DoubleBufferedBatchMetadata,
    ShapeBucket,
    ShapeBucketSelector,
    TokenizerWorker,
)
from nano_serve.runtime.overlap import BatchMetadata


@dataclass(frozen=True)
class Phase10OverlapGraphBenchmarkConfig:
    output_dir: Path
    batch_size: int = 4
    hidden_size: int = 512
    decode_steps: int = 256
    bucket_batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    bucket_seq_lens: tuple[int, ...] = (1, 2, 4, 8)
    enable_torch_compile: bool = True
    enable_cuda_graph: bool = True
    seed: int = 0
    enable_nvtx: bool = False


class _WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(item) for item in text.split()]


def run_phase10_overlap_graph_benchmark(
    config: Phase10OverlapGraphBenchmarkConfig,
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
    nsys_path = run_dir / "nsys_profile_command.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    engine_config = EngineConfig(
        graph="none",
        benchmark=BenchmarkConfig(enable_nvtx=config.enable_nvtx),
    )
    run_config = {
        "run_id": run_id,
        "phase": "phase10",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "batch_size": config.batch_size,
        "hidden_size": config.hidden_size,
        "decode_steps": config.decode_steps,
        "bucket_batch_sizes": list(config.bucket_batch_sizes),
        "bucket_seq_lens": list(config.bucket_seq_lens),
        "enable_torch_compile": config.enable_torch_compile,
        "enable_cuda_graph": config.enable_cuda_graph,
        "seed": config.seed,
        "enable_nvtx": config.enable_nvtx,
        "device": str(device),
        "dtype": str(dtype),
        "engine_config": engine_config.to_dict(),
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    nsys_path.write_text(_nsys_command(config), encoding="utf-8")

    cases: list[dict[str, object]] = []
    start_ns = time.monotonic_ns()
    with (
        JSONLEventWriter(events_path) as writer,
        nvtx_range(nvtx_label("phase10", "run"), enabled=config.enable_nvtx),
    ):
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase10"}))
        writer.write(platform_event(platform_info))
        with nvtx_range(
            nvtx_label("phase10", "cpu_overlap_primitives"),
            enabled=config.enable_nvtx,
        ):
            _run_cpu_overlap_primitives(config, writer=writer)
        selector = ShapeBucketSelector(
            [
                ShapeBucket(batch_size=batch_size, seq_len=seq_len)
                for batch_size in config.bucket_batch_sizes
                for seq_len in config.bucket_seq_lens
            ]
        )
        selection = selector.select(batch_size=config.batch_size, seq_len=1)
        model = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False),
        ).to(device=device, dtype=dtype)
        model.eval()
        generator = torch.Generator(device=device).manual_seed(config.seed)
        inputs = torch.randn(
            (selection.bucket.batch_size, selection.bucket.seq_len, config.hidden_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        selection_dict = selection.to_dict()
        case_runners = (
            ("eager", _run_eager_case),
            ("torch_compile", _run_torch_compile_case),
            ("cuda_graph", _run_cuda_graph_case),
        )
        for case_name, run_case in case_runners:
            with nvtx_range(
                nvtx_label("phase10", "case", case=case_name),
                enabled=config.enable_nvtx,
            ):
                cases.append(
                    run_case(
                        model,
                        inputs,
                        config=config,
                        device=device,
                        selection=selection_dict,
                    )
                )
        for case in cases:
            writer.write(Event("phase10_graph_case", fields=case))
        end_ns = time.monotonic_ns()
        summary = {
            "run_id": run_id,
            "phase": "phase10",
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": (end_ns - start_ns) / 1_000_000_000,
            "workload": "small_batch_decode_graphs",
            "scheduler": "continuous",
            "kv_cache": "paged_prefix",
            "cases": cases,
            "best_latency_case": min(
                [case for case in cases if case["status"] == "ok"],
                key=lambda item: _float_metric(item, "latency_ms"),
            ),
            "engine_config": engine_config.to_dict(),
            "platform": platform_info.to_dict(),
            "artifacts": {
                "run_config": str(run_config_path),
                "events": str(events_path),
                "summary": str(summary_path),
                "report": str(report_path),
                "nsys_profile_command": str(nsys_path),
            },
        }
        writer.write(Event("run_end", fields=summary))

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report_path, summary)
    return summary


def _run_cpu_overlap_primitives(
    config: Phase10OverlapGraphBenchmarkConfig,
    *,
    writer: JSONLEventWriter,
) -> None:
    tokenizer_worker = TokenizerWorker(_WhitespaceTokenizer(), max_workers=2)
    tokenizer_futures = [
        tokenizer_worker.submit(index=index, text=f"request {index} small decode")
        for index in range(config.batch_size)
    ]
    for future in tokenizer_futures:
        result = future.result(timeout=10)
        writer.write(Event("tokenizer_worker_task", fields=result.to_dict()))
    tokenizer_worker.shutdown()

    scheduler_prep = AsyncSchedulerPrep(max_workers=1)
    prep = scheduler_prep.submit(
        iteration=0,
        request_ids=[f"req-{index}" for index in range(config.batch_size)],
    ).result(timeout=10)
    writer.write(Event("async_scheduler_prep", fields=prep.to_dict()))
    scheduler_prep.shutdown()

    double_buffer = DoubleBufferedBatchMetadata()
    for iteration in range(2):
        published = double_buffer.publish(
            BatchMetadata(
                iteration=iteration,
                request_ids=tuple(f"req-{index}" for index in range(config.batch_size)),
                batch_size=config.batch_size,
                seq_len=1,
            )
        )
        writer.write(Event("double_buffer_publish", fields=published.to_dict()))


def _run_eager_case(
    model: Any,
    inputs: Any,
    *,
    config: Phase10OverlapGraphBenchmarkConfig,
    device: Any,
    selection: dict[str, object],
) -> dict[str, object]:
    latency_ms = _time_model(model, inputs, repeats=config.decode_steps, device=device)
    return _case_result(
        "eager",
        status="ok",
        latency_ms=latency_ms,
        graph_replay_count=0,
        fallback_reason=None,
        config=config,
        selection=selection,
    )


def _run_torch_compile_case(
    model: Any,
    inputs: Any,
    *,
    config: Phase10OverlapGraphBenchmarkConfig,
    device: Any,
    selection: dict[str, object],
) -> dict[str, object]:
    if not config.enable_torch_compile:
        return _skipped_case("torch_compile", "disabled", config=config, selection=selection)
    import torch

    if not hasattr(torch, "compile"):
        return _skipped_case(
            "torch_compile",
            "torch.compile is unavailable",
            config=config,
            selection=selection,
        )
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        latency_ms = _time_model(compiled, inputs, repeats=config.decode_steps, device=device)
    except Exception as exc:  # pragma: no cover - backend dependent
        return _skipped_case(
            "torch_compile",
            f"torch.compile failed: {exc}",
            config=config,
            selection=selection,
        )
    return _case_result(
        "torch_compile",
        status="ok",
        latency_ms=latency_ms,
        graph_replay_count=0,
        fallback_reason=None,
        config=config,
        selection=selection,
    )


def _run_cuda_graph_case(
    model: Any,
    inputs: Any,
    *,
    config: Phase10OverlapGraphBenchmarkConfig,
    device: Any,
    selection: dict[str, object],
) -> dict[str, object]:
    if not config.enable_cuda_graph:
        return _skipped_case("cuda_graph", "disabled", config=config, selection=selection)
    if getattr(device, "type", None) != "cuda":
        return _skipped_case("cuda_graph", "CUDA is unavailable", config=config, selection=selection)
    import torch

    try:
        graph = torch.cuda.CUDAGraph()
        static_inputs = inputs.clone()
        for _ in range(3):
            model(static_inputs)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            static_output = model(static_inputs)
        del static_output
        torch.cuda.synchronize()
        start_ns = time.monotonic_ns()
        for _ in range(config.decode_steps):
            graph.replay()
        torch.cuda.synchronize()
        latency_ms = (time.monotonic_ns() - start_ns) / 1_000_000 / config.decode_steps
    except Exception as exc:  # pragma: no cover - CUDA backend dependent
        return _skipped_case(
            "cuda_graph",
            f"CUDA graph failed: {exc}",
            config=config,
            selection=selection,
        )
    return _case_result(
        "cuda_graph",
        status="ok",
        latency_ms=latency_ms,
        graph_replay_count=config.decode_steps,
        fallback_reason=None,
        config=config,
        selection=selection,
    )


def _time_model(model: Any, inputs: Any, *, repeats: int, device: Any) -> float:
    import torch

    with torch.inference_mode():
        for _ in range(3):
            model(inputs)
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize()
        start_ns = time.monotonic_ns()
        for _ in range(repeats):
            model(inputs)
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize()
    return (time.monotonic_ns() - start_ns) / 1_000_000 / repeats


def _case_result(
    name: str,
    *,
    status: str,
    latency_ms: float | None,
    graph_replay_count: int,
    fallback_reason: str | None,
    config: Phase10OverlapGraphBenchmarkConfig,
    selection: dict[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "latency_ms": latency_ms,
        "graph_replay_count": graph_replay_count,
        "estimated_kernel_launches": config.decode_steps
        if name in {"eager", "torch_compile"}
        else 1,
        "fallback_reason": fallback_reason,
        "decode_steps": config.decode_steps,
        "hidden_size": config.hidden_size,
        **selection,
    }


def _skipped_case(
    name: str,
    reason: str,
    *,
    config: Phase10OverlapGraphBenchmarkConfig,
    selection: dict[str, object],
) -> dict[str, object]:
    return _case_result(
        name,
        status="skipped",
        latency_ms=None,
        graph_replay_count=0,
        fallback_reason=reason,
        config=config,
        selection=selection,
    )


def _nsys_command(config: Phase10OverlapGraphBenchmarkConfig) -> str:
    command = (
        "nsys profile --trace=cuda,nvtx,osrt --stats=true "
        "python3 main.py phase10-overlap-graphs "
        f"--batch-size {config.batch_size} "
        f"--hidden-size {config.hidden_size} "
        f"--decode-steps {config.decode_steps}"
    )
    if config.enable_nvtx:
        command += " --enable-nvtx"
    return command


def _float_metric(case: dict[str, object], name: str) -> float:
    value = case[name]
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _validate_config(config: Phase10OverlapGraphBenchmarkConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    if not config.bucket_batch_sizes or any(size <= 0 for size in config.bucket_batch_sizes):
        raise ValueError("bucket_batch_sizes must contain positive values")
    if not config.bucket_seq_lens or any(size <= 0 for size in config.bucket_seq_lens):
        raise ValueError("bucket_seq_lens must contain positive values")


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
