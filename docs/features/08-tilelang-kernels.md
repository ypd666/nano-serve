# TileLang Kernels

## Goal

Replace selected PyTorch operators with TileLang kernels, starting from simple
ops and paged decode attention.

## Why It Exists

The project should eventually teach kernel-level tradeoffs, but only after
system-level benchmarks can explain end-to-end impact.

## Dependencies

- Torch references.
- Paged attention reference.
- Benchmark and profiler infrastructure.

## Interfaces

- `kernels.tilelang`
- `TilePagedAttention`
- kernel microbenchmark registry
- correctness tolerance helpers
- `EngineConfig.attention_backend = "tile_paged"`
- CLI benchmark:
  `nano-serve phase7-kernels`

## Metrics

- kernel latency,
- achieved bandwidth,
- estimated FLOPs/s,
- SM activity,
- HBM bandwidth,
- occupancy,
- end-to-end TPOT impact.
- TileLang availability and skip reason.
- JSONL events:
  - `tilelang_availability`,
  - `tilelang_kernel_case`.

## Tests

- RMSNorm vs torch,
- RoPE vs torch,
- SiLU-mul vs torch,
- paged decode attention vs torch gather reference,
- dtype tolerance.

## Current Implementation Slice

Phase 7 currently provides the integration harness, correctness reference path,
and benchmark/profiling artifacts needed to develop real TileLang kernels
without breaking CPU-only or Windows development.

The `kernels.tilelang` package exposes guarded entrypoints for RMSNorm, RoPE,
SiLU-mul, sampling filter, and paged decode attention. These entrypoints keep a
torch fallback and report TileLang availability. `TilePagedAttention` exposes
the future `tile_paged` attention backend while delegating to the Phase 6 torch
gather reference until real TileLang kernels are implemented.

The local Windows environment can install the TileLang wheel, but TileLang
import currently fails during TVM DLL initialization. `phase7-kernels
--require-tilelang` therefore emits a reproducible `skipped` artifact instead
of silently benchmarking the torch fallback.

`scripts/phase7_remote_tilelang.py` is the explicit-target remote runner for
Linux NVIDIA machines. It requires `--host user@host` and does not inspect local
SSH configuration. The script clones or updates the repository on the remote
host, creates the uv environment with `torch`, `tilelang`, and `dev` extras, and
runs `phase7-kernels --require-tilelang`. With `--fetch-dir`, it copies the
remote run directory back through `scp -r`.

On H100 Linux, the TileLang import path is validated with the repository-local
uv environment. `apache-tvm-ffi` is pinned to `0.1.11` because `0.1.12` can abort
at import time with a duplicate `__ffi_repr__` type-attribute registration error.

This slice is not the final Phase 7 performance milestone. The final milestone
still requires a real TileLang paged decode attention kernel, a benchmark where
it beats the torch gather reference for at least one documented shape, and NCU
profiling on a Linux NVIDIA machine.

## Benchmarks

- shape sweep,
- dtype sweep,
- block size sweep,
- NCU profile,
- end-to-end ablation.

Examples:

```bash
nano-serve phase7-kernels \
  --hidden-size 512 \
  --seq-len 128 \
  --batch-size 2 \
  --query-heads 8 \
  --kv-heads 2 \
  --head-dim 64 \
  --context-len 512 \
  --block-size 16 \
  --repeats 10
```

Require a real TileLang kernel run, producing a skipped artifact if the
environment or implementation is not ready:

```bash
nano-serve phase7-kernels --require-tilelang
```

Remote H100/H20 runner:

```bash
python scripts/phase7_remote_tilelang.py \
  --host user@h100 \
  --remote-dir ~/nano-serve \
  --fetch-dir runs/phase7-remote
```

## Exit Criteria

- The first TileLang paged decode attention kernel beats the torch gather
  reference for at least one documented shape.
- Correctness and profiler artifacts are checked in as benchmark outputs or
  reproducible instructions.

## References

- TileLang docs.
- FlashAttention.
- Nsight Compute.

