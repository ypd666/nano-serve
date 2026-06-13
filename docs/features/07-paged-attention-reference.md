# Paged Attention Reference

## Goal

Implement a torch gather-based paged attention backend as the correctness
reference for custom kernels.

## Why It Exists

Paged KV changes memory layout. Before writing TileLang kernels, the project
needs a slow but trustworthy reference that reads block tables correctly.

## Dependencies

- Paged KV cache.

## Interfaces

- `AttentionBackend`
- `TorchGatherPagedAttention`
- block-table gather helper
- `EngineConfig.attention_backend = "torch_gather_paged"`
- CLI benchmark:
  `nano-serve phase6-attention`

## Metrics

- gather time,
- attention time,
- temporary memory,
- context length,
- block size,
- TPOT impact.
- maximum absolute difference vs contiguous attention.
- JSONL event:
  - `paged_attention_case`.

## Tests

- output vs contiguous attention,
- block boundary cases,
- GQA/MQA cases,
- long context,
- mixed sequence lengths.

## Current Implementation Slice

Phase 6 implements a torch reference for decode attention over paged KV blocks.
`TorchGatherPagedAttention.gather_kv` reconstructs per-request contiguous K/V
tensors from block tables, then `forward_decode` runs the same torch attention
math used by the contiguous reference path. This implementation is intentionally
simple and allocates temporary gathered K/V tensors; it is the correctness and
overhead baseline for later TileLang paged decode attention.

The reference supports GQA/MQA by repeating KV heads to the query-head count and
masks padded gather positions for mixed sequence lengths. Causal prefill remains
available as a contiguous reference helper.

The benchmark records `attention_backend="torch_gather_paged"` and
`kv_cache="paged"` in `engine_config`. Each sweep case emits
`paged_attention_case` with gather time, attention time, temporary gather bytes,
block size, context length, and max absolute diff against contiguous attention.

## Benchmarks

- block size sweep,
- context length sweep,
- batch size sweep,
- gather overhead isolation.

Example:

```bash
nano-serve phase6-attention \
  --batch-size 2 \
  --query-heads 8 \
  --kv-heads 2 \
  --head-dim 64 \
  --context-lens 128,512,1024 \
  --block-sizes 8,16,32 \
  --repeats 5
```

## Exit Criteria

- Correctness is stable across shapes needed by TileLang kernels.
- The overhead baseline is recorded for future speedup claims.

## References

- PagedAttention.
- FlashAttention IO-awareness for later kernel design.

