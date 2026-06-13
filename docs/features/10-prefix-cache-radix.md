# Prefix Cache and Radix Cache

## Goal

Reuse KV cache blocks for requests sharing prompt prefixes.

## Why It Exists

Multi-turn chat, RAG, and agent workloads often reuse system prompts, tool
definitions, or retrieved context. Prefix cache reduces duplicate prefill work
and improves TTFT.

## Dependencies

- Paged KV cache.
- Tokenizer-stable prompts.
- Block ref counts.

## Interfaces

- block-level prefix hash cache,
- radix tree cache,
- cache lookup/insert/evict,
- copy-on-write for shared blocks.
- `EngineConfig.kv_cache="paged_prefix"`,
- `PrefixCache.lookup(token_ids)`,
- `PrefixCache.insert(token_ids, block_ids)`,
- `PagedKVCache.allocate_prefill_with_prefix(...)`,
- `PagedKVCache.fork_request(...)`.

Phase 9 uses token-id tuples as the stable prefix identity. Full blocks are
eligible for reuse. Partial trailing blocks are not shared because divergent
decode would require copy-on-write inside a block; instead, the unmatched
partial suffix is allocated privately for the request.

`PrefixCache` maintains two indexes:

- a block hash table keyed by exact full-block token tuples,
- a compact radix tree keyed by token spans for longest-prefix lookup.

The paged cache owns block tables and ref counts. Prefix hits increment the
refcount of shared blocks. When a request appends into a shared tail block,
`PagedKVCache.allocate_decode_slot` performs copy-on-write and gives the request
a private block before mutation. Eviction is LRU over cached prefix entries and
only releases cache-owned references; live request references remain valid.

## Metrics

- prefix hit tokens,
- prefix hit rate,
- saved prefill tokens,
- TTFT improvement,
- cache memory bytes,
- matching CPU overhead.
- shared block count,
- COW copy count,
- eviction count.

## Tests

- exact token prefix matching,
- partial block behavior,
- ref count correctness,
- LRU eviction,
- copy-on-write on divergence,
- tokenizer template stability.
- benchmark JSONL schema for prefix hit/miss/eviction events.

## Benchmarks

- shared system prompt,
- multi-turn chat,
- RAG with common documents,
- cache size sweep,
- eviction stress.

Phase 9 adds a deterministic prefix-cache benchmark:

```bash
python -m nano_serve.cli phase9-prefix-cache \
  --requests 64 \
  --shared-prefix-tokens 512 \
  --unique-suffix-tokens 64 \
  --block-size 16 \
  --cache-blocks 4096
```

The benchmark runs the same synthetic workload twice:

- `paged`: no reuse, every request allocates its full prompt;
- `paged_prefix`: shared full prefix blocks are looked up and ref-counted.

It emits `prefix_cache_lookup`, `prefix_cache_insert`,
`prefix_cache_evict`, `prefix_cache_request_end`, and `prefix_cache_case`
events. Summaries report hit tokens, saved prefill tokens, hit rate, used/free
blocks, shared blocks, COW copies, evictions, and estimated TTFT improvement
from skipped prefill tokens.

## Exit Criteria

- Prefix cache improves TTFT on shared-prefix workloads.
- No incorrect reuse occurs under token/template changes.

## References

- SGLang / RadixAttention.
- PagedAttention KV sharing.

