# CPU/GPU Overlap and Graphs

## Goal

Reduce CPU scheduler/tokenizer/sampler overhead and kernel launch overhead.

## Why It Exists

Once batching and KV cache are stable, small-batch decode can be limited by CPU
gaps and launch overhead. Overlap and CUDA graphs should make those gaps
measurable and smaller.

## Dependencies

- Continuous batching.
- Benchmark profiler infrastructure.
- Stable batch metadata.

## Interfaces

- tokenizer worker,
- async scheduler preparation,
- double-buffered batch metadata,
- `torch.compile` experiment,
- CUDA graph capture with shape buckets.
- `ShapeBucketSelector`,
- `DoubleBufferedBatchMetadata`,
- `AsyncSchedulerPrep`,
- `TokenizerWorker`,
- `EngineConfig.graph="none" | "torch_compile" | "cuda_graph"`.

Phase 10 keeps all CUDA-specific behavior behind benchmark-time availability
checks. Shared code can construct bucket selectors, metadata buffers, tokenizer
workers, and async scheduler preparation without importing CUDA-only packages.
Graph execution is optional:

- `none`: eager torch reference path,
- `torch_compile`: guarded `torch.compile(..., mode="reduce-overhead")`,
- `cuda_graph`: fixed-shape CUDA graph capture/replay with shape buckets.

If `torch.compile` or CUDA graphs are unavailable, the benchmark records a
`skipped` case with the reason instead of changing default engine behavior.

## Metrics

- CPU scheduling time,
- tokenizer time,
- GPU idle gaps,
- kernel launch count,
- graph replay count,
- TPOT tail.
- selected shape bucket,
- padded batch/hidden tokens,
- metadata publish slot,
- graph fallback reason.

## Tests

- shape bucket selection,
- graph fallback behavior,
- event order under async scheduling,
- correctness with and without graph mode.
- tokenizer worker result ordering,
- double-buffered metadata alternation,
- JSONL schema for overlap/graph benchmark events.

## Benchmarks

- small batch decode,
- stable-shape decode,
- nsys timeline comparison,
- CPU-heavy online workload.

Phase 10 adds a guarded graph microbenchmark:

```bash
python -m nano_serve.cli phase10-overlap-graphs \
  --batch-size 4 \
  --hidden-size 512 \
  --decode-steps 256
```

The benchmark emits `tokenizer_worker_task`, `async_scheduler_prep`,
`double_buffer_publish`, `phase10_graph_case`, and `run_end` events. It runs an
eager torch baseline, optionally a `torch.compile` case, and optionally a CUDA
graph replay case. Each case records latency, graph replay count, estimated
kernel launches, selected shape bucket, padded elements, and fallback reason.
Every run writes an `nsys_profile_command.txt` artifact for Linux NVIDIA
profiling; H100 validation should run that command and preserve the generated
Nsight Systems report under `runs/phase10-*`.

## Exit Criteria

- Profiler timeline shows reduced GPU idle gap for a documented workload.
- Graph mode is optional and has safe fallback.

## References

- vLLM and SGLang runtime docs.
- Nsight Systems/Nsight Compute.

