from __future__ import annotations

from pathlib import Path

from nano_serve.benchmark.phase7 import (
    Phase7KernelBenchmarkConfig,
    run_phase7_kernel_benchmark,
)
from nano_serve.kernels import torch_ops
from nano_serve.kernels.tilelang import rmsnorm, rope, sample, silu_mul
from nano_serve.kernels.tilelang.availability import (
    TileLangAvailability,
    check_tilelang_available,
)
from nano_serve.observability import read_jsonl_events
from scripts.phase7_remote_tilelang import main as remote_main


def test_tilelang_wrappers_match_torch_references() -> None:
    import torch

    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    weight = torch.randn(8)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    cos = torch.randn(1, 4, 8)
    sin = torch.randn(1, 4, 8)
    up = torch.randn(2, 4, 8)
    logits = torch.randn(16)

    torch.testing.assert_close(rmsnorm(x, weight), torch_ops.rmsnorm(x, weight))

    actual_q, actual_k = rope(q, k, cos, sin)
    expected_q, expected_k = torch_ops.rope(q, k, cos, sin)
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)

    torch.testing.assert_close(silu_mul(x, up), torch_ops.silu_mul(x, up))
    torch.testing.assert_close(
        sample(logits, top_k=4, top_p=0.8),
        torch_ops.top_k_top_p_filter(logits, top_k=4, top_p=0.8),
    )


def test_phase7_kernel_benchmark_writes_case_events(tmp_path: Path) -> None:
    summary = run_phase7_kernel_benchmark(
        Phase7KernelBenchmarkConfig(
            output_dir=tmp_path / "phase7-runs",
            hidden_size=16,
            seq_len=4,
            batch_size=1,
            query_heads=2,
            kv_heads=1,
            head_dim=4,
            context_len=4,
            block_size=2,
            repeats=1,
            seed=0,
        )
    )

    assert summary["status"] == "ok"
    assert summary["phase"] == "phase7"
    assert summary["max_abs_diff"] == 0.0
    assert "tilelang_availability" in summary
    engine_config = summary["engine_config"]
    assert isinstance(engine_config, dict)
    assert engine_config["attention_backend"] == "tile_paged"

    events = read_jsonl_events(Path(summary["artifacts"]["events"]))
    case_names = [
        event["fields"]["name"]
        for event in events
        if event["name"] == "tilelang_kernel_case"
    ]
    assert case_names == [
        "rmsnorm",
        "rope",
        "silu_mul",
        "sampling_filter",
        "paged_decode_attention",
    ]
    assert any(event["name"] == "tilelang_availability" for event in events)
    assert events[-1]["name"] == "run_end"


def test_phase7_kernel_benchmark_skips_when_tilelang_required_and_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "nano_serve.benchmark.phase7.check_tilelang_available",
        lambda: TileLangAvailability(available=False, error="missing test tilelang"),
    )

    summary = run_phase7_kernel_benchmark(
        Phase7KernelBenchmarkConfig(
            output_dir=tmp_path / "phase7-runs",
            hidden_size=16,
            seq_len=4,
            batch_size=1,
            query_heads=2,
            kv_heads=1,
            head_dim=4,
            context_len=4,
            block_size=2,
            repeats=1,
            require_tilelang=True,
        )
    )

    assert summary["status"] == "skipped"
    assert summary["tilelang_available"] is False
    assert "missing test tilelang" in str(summary["skip_reason"])
    assert Path(summary["artifacts"]["ncu_profile_command"]).exists()


def test_tilelang_availability_probe_is_jsonable() -> None:
    availability = check_tilelang_available()
    payload = availability.to_dict()

    assert isinstance(payload["available"], bool)
    assert payload["probe_mode"] in {"subprocess", "in_process"}
    assert set(payload) == {"available", "version", "error", "probe_mode"}


def test_phase7_remote_runner_dry_run(capsys) -> None:
    exit_code = remote_main(
        [
            "--host",
            "user@h100",
            "--remote-dir",
            "~/nano-serve-test",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "ssh user@h100" in output
    assert "phase7-kernels --require-tilelang" in output
