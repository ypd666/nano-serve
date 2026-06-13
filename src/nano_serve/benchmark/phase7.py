"""Phase 7 TileLang kernel benchmark harness."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nano_serve.attention import TilePagedAttention, TorchGatherPagedAttention
from nano_serve.benchmark.phase6 import _pack_contiguous
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import EngineConfig
from nano_serve.kernels import torch_ops
from nano_serve.kernels.tilelang import (
    check_tilelang_available,
    rmsnorm as tile_rmsnorm,
    rope as tile_rope,
    sample as tile_sample,
    silu_mul as tile_silu_mul,
)
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform


@dataclass(frozen=True)
class Phase7KernelBenchmarkConfig:
    output_dir: Path
    hidden_size: int = 512
    seq_len: int = 128
    batch_size: int = 2
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    context_len: int = 512
    block_size: int = 16
    repeats: int = 10
    seed: int = 0
    require_tilelang: bool = False
    enable_ncu: bool = False


TILELANG_KERNELS_IMPLEMENTED = False


def run_phase7_kernel_benchmark(config: Phase7KernelBenchmarkConfig) -> dict[str, object]:
    _validate_config(config)
    import torch

    platform_info = detect_platform()
    availability = check_tilelang_available()
    run_id = _run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    profile_path = run_dir / "ncu_profile_command.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    engine_config = EngineConfig(
        kv_cache="paged",
        attention_backend="tile_paged",
        block_size=config.block_size,
    )
    run_config = {
        "run_id": run_id,
        "phase": "phase7",
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "hidden_size": config.hidden_size,
        "seq_len": config.seq_len,
        "batch_size": config.batch_size,
        "query_heads": config.query_heads,
        "kv_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "context_len": config.context_len,
        "block_size": config.block_size,
        "repeats": config.repeats,
        "seed": config.seed,
        "require_tilelang": config.require_tilelang,
        "enable_ncu": config.enable_ncu,
        "device": str(device),
        "dtype": str(dtype),
        "engine_config": engine_config.to_dict(),
        "tilelang_availability": availability.to_dict(),
        "tilelang_kernels_implemented": TILELANG_KERNELS_IMPLEMENTED,
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    profile_path.write_text(_ncu_command(config), encoding="utf-8")

    cases: list[dict[str, object]] = []
    start_ns = time.monotonic_ns()
    with JSONLEventWriter(events_path) as writer:
        writer.write(Event("run_start", fields={"run_id": run_id, "phase": "phase7"}))
        writer.write(platform_event(platform_info))
        writer.write(Event("tilelang_availability", fields=availability.to_dict()))
        skip_reason = _tilelang_skip_reason(
            require_tilelang=config.require_tilelang,
            tilelang_available=availability.available,
            availability_error=availability.error,
        )
        if skip_reason is not None:
            end_ns = time.monotonic_ns()
            summary = _summary(
                run_id=run_id,
                run_dir=run_dir,
                elapsed_s=(end_ns - start_ns) / 1_000_000_000,
                status="skipped",
                skip_reason=skip_reason,
                config=config,
                engine_config=engine_config,
                platform_info=platform_info.to_dict(),
                availability=availability.to_dict(),
                cases=cases,
                artifacts={
                    "run_config": str(run_config_path),
                    "events": str(events_path),
                    "summary": str(summary_path),
                    "report": str(report_path),
                    "ncu_profile_command": str(profile_path),
                },
            )
            writer.write(Event("run_end", fields=summary))
        else:
            generator = torch.Generator(device=device).manual_seed(config.seed)
            cases.extend(
                [
                    _rmsnorm_case(config, generator=generator, device=device, dtype=dtype),
                    _rope_case(config, generator=generator, device=device, dtype=dtype),
                    _silu_mul_case(config, generator=generator, device=device, dtype=dtype),
                    _sampling_case(config, generator=generator, device=device),
                    _paged_attention_case(config, generator=generator, device=device, dtype=dtype),
                ]
            )
            for case in cases:
                writer.write(Event("tilelang_kernel_case", fields=case))
            end_ns = time.monotonic_ns()
            summary = _summary(
                run_id=run_id,
                run_dir=run_dir,
                elapsed_s=(end_ns - start_ns) / 1_000_000_000,
                status="ok",
                skip_reason=None,
                config=config,
                engine_config=engine_config,
                platform_info=platform_info.to_dict(),
                availability=availability.to_dict(),
                cases=cases,
                artifacts={
                    "run_config": str(run_config_path),
                    "events": str(events_path),
                    "summary": str(summary_path),
                    "report": str(report_path),
                    "ncu_profile_command": str(profile_path),
                },
            )
            writer.write(Event("run_end", fields=summary))

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report_path, summary)
    return summary


def _rmsnorm_case(config: Phase7KernelBenchmarkConfig, *, generator: Any, device: Any, dtype: Any) -> dict[str, object]:
    import torch

    x = torch.randn(
        (config.batch_size, config.seq_len, config.hidden_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    weight = torch.randn((config.hidden_size,), generator=generator, device=device, dtype=dtype)
    expected = torch_ops.rmsnorm(x, weight, eps=1e-6)
    actual, latency_ms = _time_repeated(
        lambda: tile_rmsnorm(x, weight, eps=1e-6),
        repeats=config.repeats,
        device=device,
    )
    return _case_result("rmsnorm", expected, actual, latency_ms, config)


def _rope_case(config: Phase7KernelBenchmarkConfig, *, generator: Any, device: Any, dtype: Any) -> dict[str, object]:
    import torch

    q = torch.randn(
        (config.batch_size, config.query_heads, config.seq_len, config.head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        (config.batch_size, config.kv_heads, config.seq_len, config.head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    rotary_dim = config.head_dim
    positions = torch.arange(config.seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype).unsqueeze(0)
    sin = emb.sin().to(dtype=dtype).unsqueeze(0)
    expected_q, expected_k = torch_ops.rope(q, k, cos, sin)
    actual, latency_ms = _time_repeated(
        lambda: tile_rope(q, k, cos, sin),
        repeats=config.repeats,
        device=device,
    )
    actual_q, actual_k = actual
    max_abs_diff = max(_max_abs_diff(expected_q, actual_q), _max_abs_diff(expected_k, actual_k))
    return {
        **_base_case("rope", latency_ms, config),
        "max_abs_diff": max_abs_diff,
    }


def _silu_mul_case(config: Phase7KernelBenchmarkConfig, *, generator: Any, device: Any, dtype: Any) -> dict[str, object]:
    import torch

    gate = torch.randn(
        (config.batch_size, config.seq_len, config.hidden_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    up = torch.randn(gate.shape, generator=generator, device=device, dtype=dtype)
    expected = torch_ops.silu_mul(gate, up)
    actual, latency_ms = _time_repeated(
        lambda: tile_silu_mul(gate, up),
        repeats=config.repeats,
        device=device,
    )
    return _case_result("silu_mul", expected, actual, latency_ms, config)


def _sampling_case(config: Phase7KernelBenchmarkConfig, *, generator: Any, device: Any) -> dict[str, object]:
    import torch

    logits = torch.randn((config.hidden_size,), generator=generator, device=device)
    expected = torch_ops.top_k_top_p_filter(logits, top_k=32, top_p=0.9)
    actual, latency_ms = _time_repeated(
        lambda: tile_sample(logits, top_k=32, top_p=0.9),
        repeats=config.repeats,
        device=device,
    )
    return _case_result("sampling_filter", expected, actual, latency_ms, config)


def _paged_attention_case(
    config: Phase7KernelBenchmarkConfig,
    *,
    generator: Any,
    device: Any,
    dtype: Any,
) -> dict[str, object]:
    import torch

    query = torch.randn(
        (config.batch_size, config.query_heads, 1, config.head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    contiguous_key = torch.randn(
        (config.batch_size, config.kv_heads, config.context_len, config.head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    contiguous_value = torch.randn(
        (config.batch_size, config.kv_heads, config.context_len, config.head_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    paged_key, paged_value, block_tables = _pack_contiguous(
        contiguous_key,
        contiguous_value,
        config.block_size,
    )
    expected, _ = TorchGatherPagedAttention().forward_decode(
        query,
        paged_key,
        paged_value,
        block_tables,
        [config.context_len] * config.batch_size,
    )
    actual, latency_ms = _time_repeated(
        lambda: TilePagedAttention().forward_decode(
            query,
            paged_key,
            paged_value,
            block_tables,
            [config.context_len] * config.batch_size,
        )[0],
        repeats=config.repeats,
        device=device,
    )
    return {
        **_case_result("paged_decode_attention", expected, actual, latency_ms, config),
        "context_len": config.context_len,
        "block_size": config.block_size,
    }


def _time_repeated(fn, *, repeats: int, device: Any) -> tuple[Any, float]:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize()
    result = None
    start_ns = time.monotonic_ns()
    for _ in range(repeats):
        result = fn()
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize()
    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
    if result is None:
        raise RuntimeError("benchmark function produced no result")
    return result, elapsed_ms / repeats


def _case_result(
    name: str,
    expected: Any,
    actual: Any,
    latency_ms: float,
    config: Phase7KernelBenchmarkConfig,
) -> dict[str, object]:
    return {
        **_base_case(name, latency_ms, config),
        "max_abs_diff": _max_abs_diff(expected, actual),
    }


def _base_case(
    name: str,
    latency_ms: float,
    config: Phase7KernelBenchmarkConfig,
) -> dict[str, object]:
    return {
        "name": name,
        "backend": "torch_fallback",
        "tilelang_available": check_tilelang_available().available,
        "tilelang_kernels_implemented": TILELANG_KERNELS_IMPLEMENTED,
        "latency_ms": latency_ms,
        "repeats": config.repeats,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "hidden_size": config.hidden_size,
        "query_heads": config.query_heads,
        "kv_heads": config.kv_heads,
        "head_dim": config.head_dim,
    }


def _max_abs_diff(expected: Any, actual: Any) -> float:
    import torch

    expected_tensor = torch.as_tensor(expected).float()
    actual_tensor = torch.as_tensor(actual).float()
    equal_infinity = torch.isinf(expected_tensor) & torch.isinf(actual_tensor)
    diff = (expected_tensor - actual_tensor).abs()
    diff = diff.masked_fill(equal_infinity, 0.0)
    return float(torch.nan_to_num(diff, nan=0.0).max().item())


def _summary(
    *,
    run_id: str,
    run_dir: Path,
    elapsed_s: float,
    status: str,
    skip_reason: str | None,
    config: Phase7KernelBenchmarkConfig,
    engine_config: EngineConfig,
    platform_info: dict[str, object],
    availability: dict[str, object],
    cases: list[dict[str, object]],
    artifacts: dict[str, str],
) -> dict[str, object]:
    max_abs_diff = max((_float_metric(case, "max_abs_diff") for case in cases), default=0.0)
    return {
        "run_id": run_id,
        "phase": "phase7",
        "status": status,
        "skip_reason": skip_reason,
        "run_dir": str(run_dir),
        "elapsed_s": elapsed_s,
        "require_tilelang": config.require_tilelang,
        "enable_ncu": config.enable_ncu,
        "tilelang_available": bool(availability["available"]),
        "tilelang_availability": availability,
        "tilelang_kernels_implemented": TILELANG_KERNELS_IMPLEMENTED,
        "engine_config": engine_config.to_dict(),
        "cases": cases,
        "max_abs_diff": max_abs_diff,
        "platform": platform_info,
        "artifacts": artifacts,
    }


def _ncu_command(config: Phase7KernelBenchmarkConfig) -> str:
    command = [
        "ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
        "--export",
        "runs/phase7-kernels/ncu-report",
        "python",
        "-m",
        "nano_serve.cli",
        "phase7-kernels",
        "--require-tilelang",
        "--output-dir",
        "runs/phase7-kernels",
        "--hidden-size",
        str(config.hidden_size),
        "--seq-len",
        str(config.seq_len),
        "--context-len",
        str(config.context_len),
        "--block-size",
        str(config.block_size),
        "--repeats",
        str(config.repeats),
    ]
    return " ".join(command) + "\n"


def _tilelang_skip_reason(
    *,
    require_tilelang: bool,
    tilelang_available: bool,
    availability_error: str | None,
) -> str | None:
    if not require_tilelang:
        return None
    if not tilelang_available:
        return availability_error or "TileLang Python package is unavailable"
    if not TILELANG_KERNELS_IMPLEMENTED:
        return "TileLang package is available, but real TileLang kernels are not implemented yet"
    return None


def _float_metric(case: dict[str, object], name: str) -> float:
    value = case[name]
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _validate_config(config: Phase7KernelBenchmarkConfig) -> None:
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.seq_len <= 0:
        raise ValueError("seq_len must be positive")
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
    if config.context_len <= 0:
        raise ValueError("context_len must be positive")
    if config.block_size <= 0:
        raise ValueError("block_size must be positive")
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
