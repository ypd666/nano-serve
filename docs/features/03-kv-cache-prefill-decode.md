# KV Cache and Prefill/Decode Split

## Goal

Split generation into prefill and decode phases, with a contiguous KV cache.

## Why It Exists

KV cache avoids recomputing historical tokens and is the core state object for
LLM serving. It also creates the memory pressure that motivates paged KV,
prefix cache, and PD disaggregation.

## Dependencies

- Torch forwarding.
- Tokenizer, sampling, streaming.

## Interfaces

- `KVCacheManager`
- `KVHandle`
- `forward_prefill`
- `forward_decode`
- per-layer K/V tensors

## Metrics

- prefill latency,
- decode latency,
- TPOT,
- KV bytes allocated,
- cache sequence length.

## Tests

- cached decode logits vs full forward logits,
- RoPE position correctness,
- multi-step decode consistency,
- prompt length boundary cases,
- KV free on finish.

## Current Implementation Slice

The first preparatory slice separates phase callbacks from token stream
callbacks while generation still uses Phase 1 full-context forwarding.
`Engine.generate()` can emit `prefill_start`, `prefill_end`,
`decode_step_start`, and `decode_step_end` phase events independently from
sampled-token stream events.

This is a metrics semantics change only. Decode still re-runs the full prompt
plus generated tokens and does not yet consume a KV cache.

## Benchmarks

- no-cache vs KV-cache,
- output length sweep,
- context length sweep,
- memory footprint.

## Exit Criteria

- Decode uses only the new token plus KV cache.
- Correctness matches full-context reference.
- Metrics distinguish prefill and decode.

## References

- PagedAttention paper background on KV cache pressure.

