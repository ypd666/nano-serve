# Continuous Batching

## Goal

Use iteration-level scheduling so finished requests leave and new requests enter
while other requests are decoding.

## Why It Exists

Serving workloads have variable prompt and output lengths. Continuous batching
keeps the GPU busier and reduces inactive slot waste.

## Dependencies

- Static batching.
- Request state machine.
- KV cache manager for the later cached path.

## Interfaces

- `ContinuousScheduler`
- `ScheduleBudget`
- `BatchPlan`
- waiting/running/finished queues
- `Engine.step()`
- `EngineConfig.scheduler = "continuous"`
- `EngineConfig.scheduler_policy = fcfs|decode_first|prefill_first`
- offline benchmark `--scheduler continuous`

## Metrics

- running request count,
- waiting request count,
- batch size timeline,
- scheduler CPU time,
- TTFT and TPOT under RPS sweep,
- GPU idle gaps.
- continuous iteration events:
  - `continuous_iteration_start/end`,
  - `continuous_request_end`.

## Tests

- deterministic FCFS admission,
- new request admission during decode,
- finished request removal,
- capacity limits,
- cancellation behavior.

## Current Implementation Slice

Phase 4 implements an offline continuous batching baseline for
`EngineConfig.scheduler = "continuous"` and `kv_cache = "none"`.

The engine keeps explicit waiting, running, and finished queues. Each
`Engine.step()` admits new waiting requests with FCFS capacity checks, builds a
batch from active running requests, runs batched full-context torch forwarding,
samples one token per selected request, and immediately removes finished
requests from the running queue. This gives an iteration-level baseline before
KV-aware continuous batching.

The supported scheduler policies are:

- `fcfs`: preserve arrival order,
- `decode_first`: schedule decode contexts before newly admitted prefill,
- `prefill_first`: schedule newly admitted prefill before decode contexts.

`max_num_seqs` limits active running requests. `max_num_batched_tokens` limits
the selected full-context token budget for each iteration.

This implementation intentionally keeps `kv_cache = "none"` so Phase 4 can
measure scheduler behavior without crossing into paged-KV ownership. Cached
continuous batching is introduced after paged KV and paged attention are stable.

## Benchmarks

- static vs continuous batching,
- variable output lengths,
- Poisson RPS sweep,
- burst workload,
- p99 TTFT/TPOT.
- static vs continuous offline ablation through
  `--scheduler static_batch|continuous`.

## Exit Criteria

- New requests can enter without restarting the batch.
- Finished requests release KV immediately.
- Mixed-length workload improves over static batching.
- Waiting/running queue sizes and scheduler CPU time are emitted in JSONL
  iteration events.

## References

- Orca: iteration-level scheduling.
- vLLM and SGLang runtime docs.

