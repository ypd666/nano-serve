# Production-Like Observability

## Goal

Add production-style metrics, traces, dashboards, and regression benchmark
artifacts.

## Why It Exists

The project is educational, but mature serving features are hard to reason
about without observability. Tail latency, queue depth, KV pressure, and worker
utilization must be visible.

## Dependencies

- Benchmark event schema.
- Engine metrics.
- Distributed metrics for later stages.

## Interfaces

- Prometheus exporter,
- request trace id,
- per-iteration timeline dump,
- NVTX serving-stage ranges controlled by `EngineConfig.benchmark.enable_nvtx`
  and disabled by default,
- per-phase benchmark NVTX ranges controlled by each phase config's
  `enable_nvtx`, also disabled by default,
- root `main.py` CLI wrapper for direct profiler invocation,
- profiler artifact registry,
- benchmark archive,
- dashboard config.

## Metrics

- request latency histograms,
- queue depth,
- running/waiting requests,
- KV usage,
- prefix cache stats,
- speculative stats,
- worker health,
- error counts.
- Nsight-visible serving stages:
  - benchmark phase run,
  - benchmark phase case/workload,
  - request,
  - scheduler,
  - iteration,
  - prefill,
  - prefill chunk,
  - decode,
  - sampling,
  - stream callback.

## Tests

- metrics endpoint schema,
- trace id propagation,
- event timeline ordering,
- NVTX helper no-op behavior when CUDA or torch is unavailable,
- NVTX range emission when CUDA NVTX is available,
- every benchmark phase config exposes `enable_nvtx`,
- every benchmark phase CLI exposes `--enable-nvtx`,
- direct `python3 main.py` CLI wrapper smoke test,
- benchmark archive metadata,
- dashboard JSON validation.

## Benchmarks

- observability overhead,
- Prometheus scrape overhead,
- tracing enabled vs disabled,
- Nsight Systems run with `EngineConfig.benchmark.enable_nvtx=true`,
- per-phase Nsight Systems runs with phase config `enable_nvtx=true`,
- ablation run with `EngineConfig.benchmark.enable_nvtx=false`,
- default non-Nsight runs with phase config `enable_nvtx=false`,
- regression benchmark CI.

## Exit Criteria

- Local runs expose enough metrics to debug feature regressions.
- Observability can be disabled for clean performance measurement.

## References

- vLLM production metrics.
- SGLang production metrics.
- OpenTelemetry concepts where adopted.
