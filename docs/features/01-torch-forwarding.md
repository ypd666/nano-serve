# Torch Forwarding

## Goal

Run `Qwen/Qwen3.5-4B` with simple PyTorch operations.

## Why It Exists

This is the correctness baseline. It should expose model internals instead of
calling `transformers.generate()` as a hidden serving engine.

The first milestone intentionally supports only `Qwen/Qwen3.5-4B`. Avoid model
zoo abstractions until this path is correct and benchmarkable.

## Dependencies

- Infrastructure.

## Interfaces

- `ModelLoader`
- `ModelRunner`
- `TorchModelRunner`
- `SamplingParams`
- offline `Engine.generate()`
- `NANO_SERVE_MODEL_PATH`

## Metrics

- full forward latency,
- TTFT for one request,
- E2E latency,
- output tokens/s,
- peak memory.

## Tests

- load config and weights,
- compare logits against Hugging Face for a short prompt,
- deterministic greedy output,
- dtype tolerance tests.

## Benchmarks

- `single_short`,
- `single_long_prefill`,
- prompt length sweep,
- dtype sweep.

## Exit Criteria

- One model can generate with greedy decoding.
- Logits are within documented tolerance vs the oracle.
- Benchmark output is reproducible.

## References

- Hugging Face model config and safetensors formats.
- Qwen/Qwen3.5-4B.
