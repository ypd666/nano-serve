# Single-Node Distributed

## Goal

Scale from one GPU to multiple GPUs on one machine.

## Why It Exists

Larger models require sharding or replication. Single-node distributed is the
lowest-risk path to learn DP, TP, PP, and EP before multi-node deployment.

## Dependencies

- Stable model runner.
- KV cache ownership.
- Benchmark and metrics.

## Interfaces

- worker process,
- local RPC/control plane,
- data-parallel router,
- tensor parallel runner,
- pipeline parallel runner,
- expert parallel runner.
- `WorkerInfo`, `Worker`, and `WorkerLifecycle` for startup/shutdown metrics,
- `LocalRPCServer` and `LocalRPCClient` for in-process control-plane tests,
- `DataParallelRouter` for deterministic replica assignment,
- `TensorParallelPlan` for column/row linear shards and all-reduce accounting,
- `PipelineParallelPlan` for microbatch stage ordering and bubble metrics,
- `ExpertParallelPlan` for token dispatch/combine and load-balance metrics,
- `DistributedModelRunner` for selecting the reference strategy from
  `EngineConfig.parallel.mode`.

Phase 13 implements single-process reference primitives first. They model rank
ownership, sharding, local routing, and communication accounting without
requiring CUDA, NCCL, multiprocessing, or real RPC in core imports. H100
benchmarks record CUDA/NCCL availability and device count, while the reference
math remains deterministic and runnable on CPU. Later phases can replace the
local RPC and communication simulator with `torch.distributed`/NCCL workers.

## Metrics

- per-rank latency,
- NCCL time,
- all-reduce bytes,
- all-to-all bytes,
- pipeline bubble,
- per-GPU memory,
- TPOT p99.
- DP replica assignment counts,
- TP max absolute difference against dense torch reference,
- shard parameter bytes and KV shard bytes,
- worker startup/shutdown counts.

## Tests

- single-rank vs multi-rank logits,
- tensor parallel shard correctness,
- pipeline stage ordering,
- EP token dispatch/combine,
- worker startup/shutdown.
- benchmark JSONL schema for DP/TP/PP/EP/worker events.

## Benchmarks

- TP degree sweep,
- PP split sweep,
- DP replica throughput,
- EP MoE load balance.

Phase 13 adds a deterministic reference benchmark:

```bash
python -m nano_serve.cli phase13-distributed \
  --world-size 4 \
  --hidden-size 512 \
  --batch-size 8 \
  --microbatches 4
```

The benchmark emits `phase13_worker_lifecycle`, `phase13_dp_case`,
`phase13_tp_case`, `phase13_pp_case`, `phase13_ep_case`, and `run_end` events.
It reports DP routing balance, TP correctness and all-reduce bytes, PP stage
ordering and bubble ratio, EP dispatch/combine correctness and all-to-all bytes,
per-rank simulated latency, per-rank memory, and H100 CUDA device metadata.

## Exit Criteria

- One model can run with at least one multi-GPU strategy.
- Communication overhead is visible in reports.
- DP, TP, PP, and EP reference paths are selectable through
  `EngineConfig.parallel.mode`.
- Phase 13 benchmark writes JSONL events for worker lifecycle and each
  distributed strategy.

## References

- TensorRT-LLM parallelism docs.
- vLLM distributed docs.
- SGLang distributed docs.

