from __future__ import annotations

import json
from pathlib import Path

import pytest

from nano_serve.benchmark.phase13 import (
    Phase13DistributedBenchmarkConfig,
    run_phase13_distributed_benchmark,
)
from nano_serve.distributed import (
    DataParallelRouter,
    ExpertParallelPlan,
    PipelineParallelPlan,
    TensorParallelPlan,
    Worker,
    WorkerInfo,
)
from nano_serve.engine.config import EngineConfig, ParallelConfig
from nano_serve.model.distributed_runner import DistributedModelRunner


torch = pytest.importorskip("torch")


def test_worker_start_stop_lifecycle_events() -> None:
    worker = Worker(WorkerInfo(worker_id="rank-0", rank=0, world_size=2))

    running = worker.start()
    stopped = worker.stop()

    assert running.state == "running"
    assert stopped.state == "stopped"
    assert [event.state for event in worker.events] == ["created", "running", "stopped"]


def test_data_parallel_router_balances_requests() -> None:
    router = DataParallelRouter(world_size=3)

    assignments = [router.route(f"req-{index}") for index in range(6)]

    assert [item.replica_rank for item in assignments] == [0, 1, 2, 0, 1, 2]
    assert router.summary()["imbalance"] == 0


def test_tensor_parallel_matches_dense_linear() -> None:
    plan = TensorParallelPlan(world_size=2)
    x = torch.randn(4, 8)
    weight = torch.randn(8, 8)

    row = plan.run_row_parallel_linear(x, weight)
    column = plan.run_column_parallel_linear(x, weight)

    assert row.max_abs_diff < 1e-5
    assert column.max_abs_diff < 1e-5
    assert row.all_reduce_bytes > 0
    assert column.all_reduce_bytes == 0
    assert plan.kv_shard_bytes(layers=2, kv_heads=4, head_dim=8, tokens=16) > 0


def test_pipeline_parallel_schedule_preserves_stage_order() -> None:
    result = PipelineParallelPlan(stages=3).schedule(microbatches=2)

    by_microbatch = {}
    for event in result.events:
        by_microbatch.setdefault(event.microbatch, []).append(event.start_step)

    assert by_microbatch[0] == [0, 1, 2]
    assert by_microbatch[1] == [1, 2, 3]
    assert result.bubble_slots == 6
    assert result.bubble_ratio > 0


def test_expert_parallel_dispatch_combines_original_order() -> None:
    tokens = torch.randn(6, 4)
    expert_ids = torch.tensor([0, 1, 2, 3, 0, 1])

    result = ExpertParallelPlan(world_size=2, num_experts=4).dispatch_and_combine(
        tokens,
        expert_ids,
    )

    assert result.max_abs_diff == 0.0
    assert result.expert_counts == (2, 2, 1, 1)
    assert result.rank_counts == (3, 3)
    assert result.all_to_all_bytes == tokens.numel() * tokens.element_size() * 2


def test_distributed_model_runner_selects_reference_mode() -> None:
    runner = DistributedModelRunner(
        EngineConfig(parallel=ParallelConfig(mode="pp", pp_size=2))
    )

    result = runner.execute(microbatches=2).to_dict()

    assert result["mode"] == "pp"
    assert result["stages"] == 2


def test_phase13_benchmark_emits_distributed_events(tmp_path: Path) -> None:
    summary = run_phase13_distributed_benchmark(
        Phase13DistributedBenchmarkConfig(
            output_dir=tmp_path,
            world_size=2,
            hidden_size=32,
            batch_size=4,
            microbatches=2,
            num_experts=4,
            seed=1,
        )
    )

    assert summary["phase"] == "phase13"
    assert summary["status"] == "ok"
    assert summary["worker_case"]["started_workers"] == 2
    assert summary["tp_case"]["max_abs_diff"] < 1e-5
    assert summary["ep_case"]["max_abs_diff"] == 0.0
    events = [
        json.loads(line)
        for line in Path(summary["artifacts"]["events"]).read_text(encoding="utf-8").splitlines()
    ]
    names = {event["name"] for event in events}
    assert "phase13_worker_lifecycle" in names
    assert "phase13_dp_case" in names
    assert "phase13_tp_case" in names
    assert "phase13_pp_case" in names
    assert "phase13_ep_case" in names
