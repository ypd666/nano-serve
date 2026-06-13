# Chunked Prefill

## Goal

Split long prompts into prefill chunks and mix those chunks with decode work so
long prefill does not monopolize an iteration.

## Why It Exists

Long prefill can stall ongoing decode requests and damage TPOT tail latency.
Chunked prefill exposes a controllable TTFT/TPOT tradeoff.

## Dependencies

- Continuous batching.
- KV cache.
- BatchPlan that can represent mixed prefill/decode work.

## Interfaces

- `prefill_cursor` in `RequestState`
- `ChunkedPrefillScheduler`
- max prefill chunk size
- decode-maximal policy
- `EngineConfig.scheduler="chunked_prefill"`
- `EngineConfig.max_prefill_chunk_tokens`
- `TorchModelRunner.prefill_chunk(...)`

The scheduler is decode-maximal: existing decode requests are selected before
new or partially-prefilled requests. Remaining token budget is filled with
prefill chunks capped by `max_prefill_chunk_tokens`. A `BatchPlan` may contain
both decode requests and prefill chunks and uses `BatchKind.MIXED` for that
case.

The engine advances `RequestState.prefill_cursor` after each successful prefill
chunk. Intermediate prefill chunks do not emit output tokens. The first output
token is sampled only when the final prompt chunk has completed, preserving the
same output semantics as unchunked prefill. For the Phase 8 torch reference,
contiguous KV can append prompt chunks by running one cached decode step per
additional prompt token after the first chunk. The `kv_cache="none"` path is a
correctness and CPU development fallback: it records chunk scheduling events and
computes final prefill logits with the full prompt.

## Metrics

- prefill chunk size,
- decode stall time,
- TTFT p90/p99,
- TPOT p90/p99,
- prefill waiting time,
- mixed batch token counts.
- mixed iteration count,
- total prefill chunks,
- chunk size vs TTFT/TPOT frontier.

## Tests

- cursor advancement,
- chunk boundary correctness,
- mixed batch state transitions,
- cancellation during partial prefill,
- logits consistency vs unchunked prefill.
- JSONL schema for chunked-prefill benchmark events.

## Benchmarks

- long prefill plus ongoing decode,
- chunk size sweep,
- RPS sweep,
- throughput-latency frontier.

Phase 8 introduces a deterministic scheduler benchmark:

```bash
python -m nano_serve.cli phase8-chunked-prefill \
  --chunk-sizes 128,512,2048 \
  --long-prompt-tokens 8192 \
  --decode-requests 8 \
  --decode-tokens-per-request 128
```

The benchmark simulates a long prompt arriving while decode requests are already
running. It records a no-chunk baseline by setting the chunk size equal to the
long prompt length, then sweeps smaller chunk sizes. Each case emits
`chunked_prefill_iteration_start`, `chunked_prefill_iteration_end`, and
`chunked_prefill_case` events. The summary includes a frontier table and, when
`matplotlib` is installed, a chunk-size-vs-latency PNG artifact.
The case metrics include p50/p90/p99 TPOT and max decode gap because long
prefill interference often appears as a small number of very large inter-token
latency spikes.

The model-level offline ablation uses the existing offline runner:

```bash
python -m nano_serve.cli phase1-offline \
  --scheduler continuous \
  --batch-size 4 \
  --max-prompt-tokens 8192

python -m nano_serve.cli phase1-offline \
  --scheduler chunked_prefill \
  --kv-cache contiguous \
  --batch-size 4 \
  --max-prompt-tokens 8192 \
  --max-prefill-chunk-tokens 512
```

## Exit Criteria

- Long prompt arrivals no longer catastrophically stall existing decode.
- The chunk size tradeoff is visible in benchmark reports.

## References

- Sarathi-Serve.

