from __future__ import annotations

import json
from pathlib import Path

import pytest

from nano_serve.advanced import (
    JSONGrammarState,
    KVQuantizer,
    LoRAAdapter,
    LoRAAdapterRegistry,
    StructuredLogitsProcessor,
    WeightQuantizer,
)
from nano_serve.benchmark.phase12 import (
    Phase12AdvancedBenchmarkConfig,
    run_phase12_advanced_benchmark,
)
from nano_serve.engine.config import AdvancedFeatureConfig, EngineConfig


torch = pytest.importorskip("torch")


def test_weight_quantizer_reports_memory_saving_and_tolerance() -> None:
    weight = torch.linspace(-1.0, 1.0, steps=4096, dtype=torch.float32).reshape(64, 64)

    int8 = WeightQuantizer(dtype="int8", axis=1).quantize(weight)
    int4 = WeightQuantizer(dtype="int4", axis=1).quantize(weight)
    int8_restored = WeightQuantizer(dtype="int8", axis=1).dequantize(int8)
    int4_restored = WeightQuantizer(dtype="int4", axis=1).dequantize(int4)

    assert int8.memory_saving_ratio > 0.65
    assert int4.memory_saving_ratio > int8.memory_saving_ratio
    assert (int8_restored - weight).abs().max().item() < 0.01
    assert (int4_restored - weight).abs().max().item() < 0.15


def test_kv_quantizer_round_trips_with_bounded_error() -> None:
    kv = torch.randn(32, 2, 16)
    for dtype in ("int8", "fp8"):
        quantizer = KVQuantizer(dtype=dtype)

        quantized = quantizer.quantize(kv)
        restored = quantizer.dequantize(quantized)

        assert quantized.memory_saving_ratio > 0.70
        if dtype == "int8":
            assert quantized.zero_point is not None
        assert (restored - kv).abs().mean().item() < 0.02


def test_lora_registry_applies_per_request_adapters_independently() -> None:
    registry = LoRAAdapterRegistry()
    registry.register(
        LoRAAdapter(
            adapter_id="a",
            a=torch.ones(4, 2),
            b=torch.ones(2, 4),
            alpha=2.0,
        )
    )
    registry.register(
        LoRAAdapter(
            adapter_id="b",
            a=torch.full((4, 2), 2.0),
            b=torch.ones(2, 4),
            alpha=2.0,
        )
    )
    x = torch.ones(3, 4)

    output = registry.apply_batch(x, ["a", "b", "a"])

    assert torch.allclose(output[0], output[2])
    assert not torch.allclose(output[0], output[1])
    assert registry.switch_count(["a", "b", "a"]) == 2


def test_structured_processor_masks_and_rejects_invalid_json_tokens() -> None:
    processor = StructuredLogitsProcessor()
    state = JSONGrammarState()
    logits = torch.arange(0, 16, dtype=torch.float32)

    masked = processor.mask_logits(logits, state)

    assert processor.accepts(state, 0)
    assert not processor.accepts(state, 5)
    assert torch.isfinite(masked[0])
    assert not torch.isfinite(masked[5])

    for token_id in [0, 4, 2, 5, 1]:
        state = processor.advance(state, token_id)

    assert state.done


def test_engine_config_exposes_phase12_feature_flags() -> None:
    config = EngineConfig(
        advanced=AdvancedFeatureConfig(
            weight_quantization="int4",
            kv_quantization="fp8",
            lora=True,
            structured_output="json_object",
        )
    ).to_dict()

    assert config["advanced"] == {
        "weight_quantization": "int4",
        "kv_quantization": "fp8",
        "lora": True,
        "structured_output": "json_object",
    }


def test_phase12_benchmark_emits_advanced_events(tmp_path: Path) -> None:
    summary = run_phase12_advanced_benchmark(
        Phase12AdvancedBenchmarkConfig(
            output_dir=tmp_path,
            hidden_size=32,
            rank=4,
            tokens=32,
            batch_size=4,
            seed=1,
        )
    )

    assert summary["phase"] == "phase12"
    assert summary["status"] == "ok"
    assert len(summary["quant_cases"]) == 4
    assert summary["lora_case"]["adapter_count"] == 2
    assert summary["structured_case"]["done"] is True
    events = [
        json.loads(line)
        for line in Path(summary["artifacts"]["events"]).read_text(encoding="utf-8").splitlines()
    ]
    names = {event["name"] for event in events}
    assert "phase12_quant_case" in names
    assert "phase12_lora_case" in names
    assert "phase12_structured_case" in names
