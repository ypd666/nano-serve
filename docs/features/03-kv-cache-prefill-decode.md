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

- `EngineConfig.kv_cache`: `none` keeps the Phase 1 full-context baseline,
  `contiguous` enables the Phase 2 cached single-request path.
- `KVCacheManager`
- `KVHandle`
- `ContiguousKVCache`
- `forward_prefill`
- `forward_decode`
- per-layer full-attention K/V tensors
- per-layer linear-attention recurrent and convolution state
- `ContiguousKVCache.write_prefill(request_id, keys, values)`
- `ContiguousKVCache.append_decode(request_id, keys, values)`
- `ContiguousKVCache.get_kv(request_id)`

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
- contiguous prefill write/read shape and value roundtrip,
- decode append increments cache sequence length,
- capacity overflow raises a clear error,
- `free()` removes request ownership,
- CPU-only tensor allocation works.

## Current Implementation Slice

Phase 2 supports two comparable execution paths:

- `kv_cache=none`: the Phase 1 baseline. Prefill and decode are separate runner
  calls, but decode re-runs the full prompt plus generated tokens.
- `kv_cache=contiguous`: prefill allocates one contiguous per-request cache and
  fills per-layer state. Decode consumes only the newest token plus cached
  state.

Qwen3.5-4B mixes full-attention and linear-attention layers. The contiguous
cache therefore stores full-attention K/V tensors for full-attention layers and
linear recurrent plus convolution state for linear-attention layers. RoPE uses
the cache sequence length as the decode token position so cached decode logits
match full-context logits.

`Engine.generate()` emits phase callbacks separately from token stream
callbacks. Phase events mark `prefill_start`, `prefill_end`,
`decode_step_start`, and `decode_step_end`; stream events only mean a sampled
token was emitted. Phase events include KV cache metadata for ablation.

### Contiguous KV Layout Slice

This slice adds a real contiguous KV cache layout before wiring it into model
  decode.

The cache stores keys and values in preallocated tensors with shape:

  ```text
  [num_layers, max_tokens, num_heads, head_dim]
  ```

Each request owns one contiguous token range. Prefill writes prompt K/V into the
  range, decode appends one token at a time, and free() releases the request
  metadata. The first allocator may use a monotonic cursor and does not need
  compaction or reuse.

This slice does not implement paged KV, prefix sharing, or TileLang attention.
  Model decode integration is a follow-up step after allocator correctness tests
  pass.

## Benchmarks

- no-cache vs KV-cache through `--kv-cache none|contiguous`,
- output length sweep,
- context length sweep,
- memory footprint from contiguous cache byte accounting.
- allocator append/free microbenchmark,
- prefill write bandwidth by prompt length,
- decode append overhead by output length.

## Exit Criteria

- Decode uses only the new token plus KV cache when `kv_cache=contiguous`.
- Correctness matches full-context reference.
- Metrics distinguish prefill and decode.
- JSONL events record KV sequence length and bytes used.

## References

- PagedAttention paper background on KV cache pressure.
