# Static Batching

## Goal

Run a fixed group of requests together through prefill and decode.

## Why It Exists

Static batching is a simple baseline that reveals padding waste and inactive
slot waste. That waste is the motivation for continuous batching.

## Dependencies

- KV cache and prefill/decode split.

## Interfaces

- `StaticBatchScheduler`
- batch padding metadata
- inactive slot tracking
- `EngineConfig.scheduler = "static_batch"`
- offline benchmark `--scheduler static_batch`
- `TorchModelRunner.next_token_logits_batch`

## Metrics

- batch size,
- padding tokens,
- inactive slots,
- effective tokens/s,
- per-request TTFT/TPOT.
- static batch iteration events:
  - `batch_prefill_start/end`,
  - `batch_decode_step_start/end`,
  - `batch_end`,
  - `batch_request_end`.

## Tests

- same-length batch correctness,
- mixed-length batch correctness,
- finished request masking,
- stop conditions per request.

## Current Implementation Slice

Phase 3 implements a fixed offline static batch baseline for
`EngineConfig.scheduler = "static_batch"` and `kv_cache = "none"`.

The engine admits a fixed group of requests together, right-pads prompt/context
tokens to the longest slot, and runs batched full-context torch forwarding for
prefill and decode. Decode keeps the batch shape fixed until all requests finish
or hit `max_tokens`; finished requests become inactive slots and still
contribute to inactive-slot waste metrics.

This is intentionally a waste-measurement baseline. It does not yet combine
static batching with the Phase 2 contiguous KV cache. KV-aware batching is left
for the scheduler/KV ownership work that follows.

## Benchmarks

- equal prompt/output lengths,
- mixed prompt/output lengths,
- batch size sweep,
- inactive slot waste.
- static batch vs single-request sequential ablation through
  `--scheduler single|static_batch`.

## Exit Criteria

- Batch shape stays fixed during generation.
- Waste metrics clearly show why static batching is insufficient.
- Per-request stop conditions are honored while fixed slots remain visible in
  batch metrics.

## References

- Orca motivation for iteration-level scheduling.

