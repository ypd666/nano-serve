# Infrastructure

## Goal

Create the project skeleton, config system, benchmark event schema, and report
pipeline before implementing model execution.

## Why It Exists

The project is an ablation platform. Without reproducible benchmark artifacts,
later features cannot be compared fairly.

## Dependencies

None.

## Interfaces

- `EngineConfig`
- `FeatureFlags`
- `RequestMetrics`
- `IterationMetrics`
- `BenchmarkRunConfig`
- JSONL event writer
- report and compare commands
- asset downloader for the first model and serving dataset
- ShareGPT dataset fixture loader
- Phase 0 local smoke runner
- platform detector for CPU-only macOS development and CUDA Linux benchmarking

## Metrics

- request-level timestamps,
- iteration batch statistics,
- system throughput,
- GPU memory placeholders,
- profiler artifact paths.
- platform fields: OS, machine, Python version, torch version if installed,
  detected device backend, CUDA device count/names when available, and macOS
  Apple Silicon flag.

## Tests

- config serialization round trip,
- event schema validation,
- event writer round trip,
- report and comparison rendering,
- import smoke test,
- environment variable parsing for local asset paths,
- dataset fixture loading,
- Phase 0 smoke artifact generation,
- gitignore check for repo-local model and dataset paths,
- device detection tests for no CUDA and CUDA-available states.

## Benchmarks

Start with a dummy workload that checks assets, reads a small ShareGPT sample,
and emits deterministic events without running a model. The benchmark harness
should work before the engine works.

## Exit Criteria

- `nano-serve phase0-smoke` and `nano-serve bench dummy` run locally.
- A run emits `run_config.json`, `events.jsonl`, `summary.json`, and `report.md`.
- `nano-serve bench compare` compares two run directories or summary files.
- Future feature flags can be represented in config.
- `scripts/download_assets.py --print-env-template` documents the required
  local asset environment variables.
- `NANO_SERVE_MODEL_PATH` and `NANO_SERVE_DATASET_PATH` point to gitignored
  local paths before downloading.
- Core Phase 0 commands run on macOS Apple Silicon without CUDA by using CPU.
- CUDA-specific paths are optional and reserved for Linux NVIDIA H20/H100
  benchmark/profiling runs.

## Platform Policy

Shared infrastructure must support:

- macOS Apple Silicon with CPU-only execution for local agent-loop development.
- Linux NVIDIA H20/H100 with CUDA for full model-loading and performance work.

Device selection is intentionally simple in Phase 0: prefer CUDA when
`torch.cuda.is_available()` is true; otherwise use CPU. Do not add an MPS path
for macOS. CUDA-only, TileLang, TensorRT, and Triton imports must stay gated
behind feature flags or local availability checks.

## References

- NVIDIA GenAI-Perf metrics.
- vLLM metrics design.
- SGLang production metrics.
- Hugging Face model and dataset repositories.
