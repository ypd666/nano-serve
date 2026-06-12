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

The first Phase 2 slice splits the no-cache generation loop into explicit
`prefill` and `decode` runner calls. Prefill computes the first next-token logits
from the prompt. Decode computes later next-token logits from the full prompt
plus generated tokens.

`Engine.generate()` now emits phase callbacks separately from token stream
callbacks. Phase events mark `prefill_start`, `prefill_end`,
`decode_step_start`, and `decode_step_end`; stream events only mean a sampled
token was emitted.

This is an interface and metrics split only. Decode still uses full-context
forwarding and does not yet consume a KV cache.

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

