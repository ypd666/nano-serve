"""Phase 13 single-node distributed benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nano_serve.benchmark.profiler import nvtx_label, nvtx_range
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.distributed import (
    DataParallelRouter,
    ExpertParallelPlan,
    PipelineParallelPlan,
    TensorParallelPlan,
    Worker,
    WorkerInfo,
)
from nano_serve.engine.config import BenchmarkConfig, EngineConfig, ParallelConfig
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform


@dataclass(frozen=True)
class Phase13DistributedBenchmarkConfig:
    output_dir: Path
    world_size: int = 4
    hidden_size: int = 512
    batch_size: int = 8
    microbatches: int = 4
    num_experts: int = 8
    seed: int = 0
    enable_nvtx: bool = False


def run_phase13_distributed_benchmark(
    config: Phase13DistributedBenchmarkConfig,
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
        parallel=ParallelConfig(
            mode="tp",
            tp_size=config.world_size,
            pp_size=config.world_size,
            dp_size=config.world_size,
            ep_size=config.world_size,
        ),
        benchmark=BenchmarkConfig(enable_nvtx=config.enable_nvtx),
    )
    run_config = {
        "run_id": run_id,
        "phase": "phase13",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "world_size": config.world_size,
        "hidden_size": config.hidden_size,
        "batch_size": config.batch_size,
        "microbatches": config.microbatches,
        "num_experts": config.num_experts,
        "seed": config.seed,
        "enable_nvtx": config.enable_nvtx,
        "engine_config": engine_config.to_dict(),
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    start_ns = time.monotonic_ns()
    with (
        JSONLEventWriter(events_path) as writer,
        nvtx_range(nvtx_label("phase13", "run"), enabled=config.enable_nvtx),
    ):
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase13"}))
        writer.write(platform_event(platform_info))
        with nvtx_range(
            nvtx_label("phase13", "case", case="worker"),
            enabled=config.enable_nvtx,
        ):
            worker_case = _run_worker_case(config, writer)
        with nvtx_range(
            nvtx_label("phase13", "case", case="dp"),
            enabled=config.enable_nvtx,
        ):
            dp_case = _run_dp_case(config, writer)
        with nvtx_range(
            nvtx_label("phase13", "case", case="tp"),
            enabled=config.enable_nvtx,
        ):
            tp_case = _run_tp_case(config, writer)
        with nvtx_range(
            nvtx_label("phase13", "case", case="pp"),
            enabled=config.enable_nvtx,
        ):
            pp_case = _run_pp_case(config, writer)
        with nvtx_range(
            nvtx_label("phase13", "case", case="ep"),
            enabled=config.enable_nvtx,
        ):
            ep_case = _run_ep_case(config, writer)
        end_ns = time.monotonic_ns()
        summary = {
            "run_id": run_id,
            "phase": "phase13",
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": (end_ns - start_ns) / 1_000_000_000,
            "workload": "single_node_distributed_reference",
            "scheduler": "continuous",
            "kv_cache": "paged_prefix",
            "worker_case": worker_case,
            "dp_case": dp_case,
            "tp_case": tp_case,
            "pp_case": pp_case,
            "ep_case": ep_case,
            "max_communication_bytes": max(
                _int_metric(tp_case, "all_reduce_bytes"),
                _int_metric(ep_case, "all_to_all_bytes"),
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


def _run_worker_case(
    config: Phase13DistributedBenchmarkConfig,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    workers = [
        Worker(
            WorkerInfo(
                worker_id=f"rank-{rank}",
                rank=rank,
                world_size=config.world_size,
                role="model",
            )
        )
        for rank in range(config.world_size)
    ]
    events = []
    for worker in workers:
        for event in (worker.start(), worker.stop()):
            event_dict = event.to_dict()
            events.append(event_dict)
            writer.write(Event("phase13_worker_lifecycle", fields=event_dict))
    case = {
        "world_size": config.world_size,
        "started_workers": sum(1 for event in events if event["state"] == "running"),
        "stopped_workers": sum(1 for event in events if event["state"] == "stopped"),
        "events": events,
    }
    return case


def _run_dp_case(
    config: Phase13DistributedBenchmarkConfig,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    router = DataParallelRouter(world_size=config.world_size)
    for request_index in range(config.batch_size * 2):
        assignment = router.route(f"req-{request_index}")
        writer.write(Event("phase13_dp_assignment", fields=assignment.to_dict()))
    case = {
        "case": "data_parallel_least_loaded",
        **router.summary(),
        "simulated_replica_throughput": config.batch_size * 2 / max(1, config.world_size),
    }
    writer.write(Event("phase13_dp_case", fields=case))
    return case


def _run_tp_case(
    config: Phase13DistributedBenchmarkConfig,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    import torch

    plan = TensorParallelPlan(world_size=config.world_size)
    x = torch.randn(config.batch_size, config.hidden_size)
    weight = torch.randn(config.hidden_size, config.hidden_size)
    row = plan.run_row_parallel_linear(x, weight)
    column = plan.run_column_parallel_linear(x, weight)
    kv_shard_bytes = plan.kv_shard_bytes(
        layers=4,
        kv_heads=max(config.world_size, 4),
        head_dim=64,
        tokens=1024,
    )
    case = {
        "case": "tensor_parallel_linear",
        "world_size": config.world_size,
        "row_parallel": row.to_dict(),
        "column_parallel": column.to_dict(),
        "max_abs_diff": max(row.max_abs_diff, column.max_abs_diff),
        "all_reduce_bytes": row.all_reduce_bytes,
        "shard_parameter_bytes": max(
            row.shard_parameter_bytes,
            column.shard_parameter_bytes,
        ),
        "kv_shard_bytes": kv_shard_bytes,
    }
    writer.write(Event("phase13_tp_case", fields=case))
    return case


def _run_pp_case(
    config: Phase13DistributedBenchmarkConfig,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    plan = PipelineParallelPlan(stages=config.world_size)
    result = plan.schedule(microbatches=config.microbatches)
    case = {"case": "pipeline_parallel_1f1b_reference", **result.to_dict()}
    for event in result.events:
        writer.write(Event("phase13_pp_stage", fields=event.to_dict()))
    writer.write(Event("phase13_pp_case", fields=case))
    return case


def _run_ep_case(
    config: Phase13DistributedBenchmarkConfig,
    writer: JSONLEventWriter,
) -> dict[str, object]:
    import torch

    plan = ExpertParallelPlan(
        world_size=config.world_size,
        num_experts=config.num_experts,
    )
    tokens = torch.randn(config.batch_size * config.microbatches, config.hidden_size)
    expert_ids = torch.arange(tokens.shape[0]) % config.num_experts
    result = plan.dispatch_and_combine(tokens, expert_ids)
    case = {"case": "expert_parallel_dispatch", **result.to_dict()}
    writer.write(Event("phase13_ep_case", fields=case))
    return case


def _validate_config(config: Phase13DistributedBenchmarkConfig) -> None:
    if config.world_size <= 0:
        raise ValueError("world_size must be positive")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.hidden_size % config.world_size != 0:
        raise ValueError("hidden_size must be divisible by world_size")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.microbatches <= 0:
        raise ValueError("microbatches must be positive")
    if config.num_experts <= 0:
        raise ValueError("num_experts must be positive")


def _int_metric(case: dict[str, object], name: str) -> int:
    value = case[name]
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


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
