# Quantization, LoRA, and Structured Output

## Goal

Add mature serving features after the scheduler, KV cache, and benchmark stack
are stable.

## Why It Exists

Quantization reduces memory/bandwidth pressure. LoRA and structured output are
common production serving features. They should be implemented as measurable
experiments, not as hidden complexity in early milestones.

## Dependencies

- Model runner abstraction.
- Attention/KV metrics.
- Quality/correctness regression tests.

## Interfaces

- quantized weight loader,
- KV quantization backend,
- LoRA adapter registry,
- multi-LoRA batching metadata,
- grammar/structured output decoder.
- `WeightQuantizer` for per-row INT8/INT4 experiments,
- `KVQuantizer` for per-tensor INT8 and simulated FP8 KV cache experiments,
- `LoRAAdapter` and `LoRAAdapterRegistry`,
- `StructuredLogitsProcessor` for constrained token masks,
- `EngineConfig.kv_cache="paged_prefix"` remains the baseline while Phase 12
  features are opt-in experiments.

Phase 12 implements reference experiments rather than replacing the Qwen weight
loader. Quantized weights are represented as packed tensors plus scale metadata,
then dequantized for correctness checks and microbenchmarks. KV quantization
uses per-tensor INT8 scale/zero-point metadata and a simulated FP8 one-byte
symmetric reference for key/value pages. LoRA is
implemented as `x @ A @ B * scale` with an adapter registry and per-request
adapter ids for multi-LoRA batching metadata. Structured output starts with a
finite-state JSON-object grammar for token-level acceptance/rejection.

## Metrics

- memory saved,
- TPOT/E2E impact,
- quality/correctness deltas,
- adapter switch overhead,
- structured decoding overhead.
- max/mean absolute error,
- quantized bytes vs baseline bytes,
- LoRA adapter count and switch count,
- grammar accepted/rejected token counts.

## Tests

- quantized output tolerance,
- KV quant decode stability,
- LoRA adapter correctness,
- multi-LoRA batching isolation,
- grammar acceptance/rejection.
- benchmark JSONL schema for quantization/LoRA/structured cases.

## Benchmarks

- memory-constrained workloads,
- long-context KV quant,
- multi-adapter mixed workload,
- JSON/grammar constrained decoding.

Phase 12 adds a deterministic feature benchmark:

```bash
python -m nano_serve.cli phase12-advanced \
  --hidden-size 512 \
  --rank 8 \
  --tokens 1024
```

The benchmark emits `phase12_quant_case`, `phase12_lora_case`,
`phase12_structured_case`, and `run_end` events. It reports INT8/INT4
weight-only memory savings and reconstruction error, INT8 and simulated FP8 KV
quant/dequant error, LoRA adapter isolation and switch count, structured grammar
accept/reject counts, and simulated overheads. These are ablation artifacts
against the float32 torch reference and remain optional through
`EngineConfig.advanced` flags.

## Exit Criteria

- Each feature has a quality and performance tradeoff report.
- Features remain optional through `EngineConfig.advanced` flags.
- Phase 12 benchmark emits quantization, LoRA, and structured-output JSONL
  events.

## References

- vLLM feature docs.
- SGLang feature docs.
- TensorRT-LLM optimization docs.

