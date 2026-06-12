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
- `TokenizerWrapper`
- `HuggingFaceOracle`
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
- load local tokenizer,
- compare oracle logits shape for a short prompt,
- compare logits against Hugging Face for a short prompt,
- deterministic greedy output,
- dtype tolerance tests.

## Current Implementation Slice

Phase 1 adds the local config reader, text-only tokenizer wrapper, Hugging Face
oracle, safetensors loader, and a narrow Qwen3.5 text-only PyTorch forwarding
path.

The current `TorchModelRunner` supports one request at a time and runs full
prompt forwarding without KV cache. It implements the Qwen3.5-4B text stack,
including the mixed linear-attention and full-attention layer pattern, then
returns logits through the shared `ModelOutput` interface.

`Engine.generate()` wires the runner into a minimal offline generation loop.
Each decode step reruns the full prompt plus generated tokens; this is the
Phase 1 baseline that later KV-cache and scheduling features ablate against.
The loop supports greedy decoding, temperature sampling, top-k/top-p filtering,
and an optional token stream callback.

The oracle is used only as a correctness reference for nano-serve torch
forwarding; it is not the serving engine.

Heavy model tests are opt-in with `NANO_SERVE_RUN_HEAVY_TESTS=1` so regular unit
tests do not load the full Qwen3.5-4B weights. The heavy tests compare the
nano-serve PyTorch logits against the Hugging Face eager oracle for a short
multi-token prompt.

Current non-goals:

- KV cache,
- batching,
- vision inputs,
- alternate model ids.

## Benchmarks

- `single_short`,
- `single_long_prefill`,
- prompt length sweep,
- dtype sweep.

The implemented Phase 1 benchmark entrypoint is:

```bash
python -m nano_serve.cli phase1-offline \
  --num-samples 1 \
  --max-new-tokens 8 \
  --max-prompt-tokens 128 \
  --output-dir runs/phase1
```

It writes `run_config.json`, `events.jsonl`, `summary.json`, and `report.md`.
The first local baseline run on the RTX 4070 SUPER generated 8 output tokens for
one ShareGPT sample with full-context greedy decoding:

- run dir: `runs/phase1/20260612T080937.484558Z`,
- input tokens: 34,
- output tokens: 8,
- TTFT: 4311.2558 ms,
- TPOT: 183.8098 ms,
- E2E: 5597.9241 ms,
- output tokens/s: 1.4263.

## Exit Criteria

- One model can generate with greedy decoding.
- Logits are within documented tolerance vs the oracle.
- Benchmark output is reproducible.

## References

- Hugging Face model config and safetensors formats.
- Qwen/Qwen3.5-4B.
