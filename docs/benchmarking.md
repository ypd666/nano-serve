# Benchmarking

Benchmarking is part of the product, not an afterthought.

## Benchmark Layers

### Microbenchmark

Measures one operator or subsystem:

- RMSNorm, RoPE, SiLU-mul.
- prefill attention and decode attention.
- paged attention.
- sampling.
- KV append/free/gather.
- TileLang kernel vs torch reference.

### Offline Throughput

Runs a fixed prompt set without simulating request arrival:

- total tokens/s,
- output tokens/s,
- input tokens/s,
- average batch size,
- GPU memory peak,
- KV utilization,
- MFU,
- SM activity,
- HBM bandwidth utilization.

Phase 2 offline ablations use the same workload with different KV cache flags:

```bash
python -m nano_serve.cli phase1-offline --kv-cache none
python -m nano_serve.cli phase1-offline --kv-cache contiguous
```

Both runs emit the same request latency metrics. The contiguous run also emits
`kv_sequence_length`, `kv_blocks_used`, `kv_bytes_used`, and
`kv_fragmentation` in phase events and request summaries.

Phase 3 static batching ablations use the same offline runner with different
schedulers:

```bash
python -m nano_serve.cli phase1-offline --scheduler single --kv-cache none
python -m nano_serve.cli phase1-offline --scheduler static_batch --batch-size 4 --kv-cache none
```

The static-batch run emits `batch_prefill_start/end`,
`batch_decode_step_start/end`, `batch_end`, and `batch_request_end` events.
Batch summaries record `total_padded_tokens`, `total_inactive_slot_steps`,
`model_invocations`, and `decode_invocations` so fixed-batch waste can be
compared against the single-request baseline.

Phase 4 continuous batching uses the same offline runner and compares against
the static batch baseline:

```bash
python -m nano_serve.cli phase1-offline --scheduler static_batch --batch-size 4 --kv-cache none
python -m nano_serve.cli phase1-offline --scheduler continuous --batch-size 4 --kv-cache none
```

The continuous run emits `continuous_iteration_start/end` and
`continuous_request_end` events. Iteration events record `batch_kind`,
`batch_size`, `num_prefill_tokens`, `num_decode_tokens`,
`num_running_reqs`, `num_waiting_reqs`, `cpu_schedule_time_ms`,
`padded_tokens`, and `request_ids`.

Phase 5 allocator benchmarks exercise paged KV without loading model weights:

```bash
python -m nano_serve.cli phase5-kv --num-blocks 128 --block-size 16
```

The paged-KV run emits `paged_kv_prefill`, `paged_kv_decode_end`,
`paged_kv_free`, and `paged_kv_oom` events. Summaries record used/free blocks,
used tokens, allocated token capacity, internal fragmentation, OOM count, and
max resident requests.

Phase 6 paged-attention benchmarks isolate the torch gather reference path:

```bash
python -m nano_serve.cli phase6-attention \
  --batch-size 2 \
  --query-heads 8 \
  --kv-heads 2 \
  --head-dim 64 \
  --context-lens 128,512,1024 \
  --block-sizes 8,16,32 \
  --repeats 5
```

The paged-attention run emits `paged_attention_case` events. Each case records
batch size, query/KV heads, head dimension, context length, block size, gather
time, attention time, temporary gather bytes, and max absolute difference
against contiguous attention. The run config records
`attention_backend="torch_gather_paged"` and `kv_cache="paged"` so later
TileLang kernels can use the same sweep as an ablation baseline.

### Online Serving

Simulates request arrival and user-observed latency:

- TTFT p50/p90/p99,
- TPOT p50/p90/p99,
- E2E p50/p90/p99,
- requests/s,
- goodput under SLO,
- queueing time,
- prefill waiting time,
- decode waiting time,
- cancellation and timeout behavior.

## Metric Definitions

`TTFT`:

```text
first_token_ts - arrival_ts
```

`TPOT` / `ITL`:

```text
(last_token_ts - first_token_ts) / (output_tokens - 1)
```

For one-token outputs, TPOT is undefined and should be recorded as null.

`E2E`:

```text
last_token_ts - arrival_ts
```

`MFU`:

```text
estimated_model_flops_per_second / theoretical_peak_flops
```

Report `mfu_prefill` and `mfu_decode` separately. Decode is often limited by
memory bandwidth and KV reads, so HBM utilization and SM activity should be
reported with MFU.

## Standard Workloads

| Workload | Input | Output | Purpose |
| --- | --- | --- | --- |
| `single_short` | 128 | 128 | correctness and simple latency |
| `single_long_prefill` | 8192 | 128 | prefill pressure |
| `decode_long` | 128 | 2048 | decode/KV pressure |
| `mixed_chat` | 256-4096 | 64-1024 | batching behavior |
| `shared_prefix` | shared 80% prefix | 128-512 | prefix cache |
| `burst` | mixed | mixed | queueing/admission |
| `poisson_rps_sweep` | mixed | mixed | online SLO curve |
| `spec_decode_friendly` | stable | 256-512 | high acceptance speculation |
| `spec_decode_hostile` | random | 256-512 | low acceptance speculation |

## Artifact Contract

Every run should produce:

- `run_config.json`,
- `events.jsonl`,
- `summary.json`,
- `report.md`,
- optional profiler artifacts.

The run config must include command, git commit, model, dtype, hardware,
feature flags, workload, random seed, and relevant package versions.

