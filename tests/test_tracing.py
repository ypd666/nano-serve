from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from nano_serve.benchmark.offline import OfflineBenchmarkConfig
from nano_serve.benchmark.phase0 import Phase0SmokeConfig
from nano_serve.benchmark.phase5 import Phase5KVBenchmarkConfig
from nano_serve.benchmark.phase6 import Phase6PagedAttentionBenchmarkConfig
from nano_serve.benchmark.phase7 import Phase7KernelBenchmarkConfig
from nano_serve.benchmark.phase8 import Phase8ChunkedPrefillBenchmarkConfig
from nano_serve.benchmark.phase9 import Phase9PrefixCacheBenchmarkConfig
from nano_serve.benchmark.phase10 import Phase10OverlapGraphBenchmarkConfig
from nano_serve.benchmark.phase11 import Phase11SpeculativeBenchmarkConfig
from nano_serve.benchmark.phase12 import Phase12AdvancedBenchmarkConfig
from nano_serve.benchmark.phase13 import Phase13DistributedBenchmarkConfig
from nano_serve.benchmark.profiler import nvtx_label
from nano_serve.observability.tracing import nvtx_range


def test_nvtx_range_uses_torch_cuda_nvtx_when_available(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    class _FakeRange:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            events.append(("enter", self.name))

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback
            events.append(("exit", self.name))

    fake_nvtx = SimpleNamespace(range=lambda name: _FakeRange(name))
    fake_cuda = SimpleNamespace(is_available=lambda: True, nvtx=fake_nvtx)
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with nvtx_range("nano_serve.prefill"):
        events.append(("body", "nano_serve.prefill"))

    assert events == [
        ("enter", "nano_serve.prefill"),
        ("body", "nano_serve.prefill"),
        ("exit", "nano_serve.prefill"),
    ]


def test_nvtx_range_is_noop_when_disabled(monkeypatch) -> None:
    events: list[str] = []
    fake_nvtx = SimpleNamespace(range=lambda name: events.append(name))
    fake_cuda = SimpleNamespace(is_available=lambda: True, nvtx=fake_nvtx)
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with nvtx_range("nano_serve.decode", enabled=False):
        events.append("body")

    assert events == ["body"]


def test_all_phase_configs_expose_nvtx_ablation_flag() -> None:
    config_types = [
        Phase0SmokeConfig,
        OfflineBenchmarkConfig,
        Phase5KVBenchmarkConfig,
        Phase6PagedAttentionBenchmarkConfig,
        Phase7KernelBenchmarkConfig,
        Phase8ChunkedPrefillBenchmarkConfig,
        Phase9PrefixCacheBenchmarkConfig,
        Phase10OverlapGraphBenchmarkConfig,
        Phase11SpeculativeBenchmarkConfig,
        Phase12AdvancedBenchmarkConfig,
        Phase13DistributedBenchmarkConfig,
    ]

    for config_type in config_types:
        config_fields = {field.name: field for field in fields(config_type)}
        assert "enable_nvtx" in config_fields
        assert config_fields["enable_nvtx"].default is False


def test_nvtx_label_formats_phase_metadata() -> None:
    assert (
        nvtx_label("phase11", "case", workload="friendly", gamma=4)
        == "nano_serve.phase11.case:gamma=4,workload=friendly"
    )


def test_main_py_cli_wrapper_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Learning-oriented LLM serving engine" in result.stdout
