# Paged KV Cache

## Goal

Manage KV cache as fixed-size blocks with per-request block tables.

## Why It Exists

Contiguous KV allocation wastes memory under variable sequence lengths and
dynamic request lifetimes. Paged KV improves memory utilization and enables
prefix sharing.

## Dependencies

- KV cache.
- Continuous batching.

## Interfaces

- `PagedKVCache`
- `KVBlock`
- `BlockTable`
- free block allocator
- ref-counted blocks
- allocator benchmark `phase5-kv`

## Metrics

- allocated blocks,
- free blocks,
- used tokens,
- internal fragmentation,
- OOM count,
- max resident requests.
- JSONL events:
  - `paged_kv_prefill`,
  - `paged_kv_decode_end`,
  - `paged_kv_free`,
  - `paged_kv_oom`.

## Tests

- randomized allocate/free,
- block table sequence length invariants,
- append-token allocation,
- OOM behavior,
- logits consistency vs contiguous KV.

## Current Implementation Slice

Phase 5 implements the paged KV allocator and block-table accounting without
hooking paged attention into model execution yet.

`PagedKVCache` owns a fixed pool of `KVBlock` records, a sorted free block
allocator, and one `BlockTable` per resident request. Prefill allocates enough
blocks for the current prompt, decode appends tokens and allocates a new block
only at block boundaries, and free releases all request blocks back to the free
list. OOM attempts increment `oom_count` and raise `MemoryError`.

Stats expose used/free blocks, used tokens, allocated token capacity, internal
fragmentation, OOM count, resident requests, and max resident requests.

Logits consistency against contiguous KV is deferred to Phase 6, where the
torch gather-based paged attention reference can consume these block tables.

## Benchmarks

- contiguous vs paged KV,
- high concurrency,
- long output,
- mixed length,
- block size sweep.
- allocator microbenchmark through `phase5-kv`.

## Exit Criteria

- Paged KV matches contiguous KV correctness.
- Fragmentation and OOM behavior are measurable.
- Block table is stable enough for paged attention.
- Allocator events and summary artifacts are emitted for benchmark comparison.

## References

- PagedAttention / vLLM paper.

