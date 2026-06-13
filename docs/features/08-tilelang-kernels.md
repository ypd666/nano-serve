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
benchmark/profiling artifacts, and the first real TileLang paged decode
attention kernel without breaking CPU-only or Windows development.

The `kernels.tilelang` package exposes guarded entrypoints for RMSNorm, RoPE,
SiLU-mul, sampling filter, and paged decode attention. These entrypoints keep a
torch fallback and report TileLang availability. `TilePagedAttention` exposes
the `tile_paged` attention backend and dispatches decode-one-token CUDA
float16 paged attention to TileLang when the shape is supported.

The first TileLang kernel set is intentionally narrow. RMSNorm, RoPE,
SiLU-mul, and the sampling filter support CUDA `float16` benchmark tensors and
fall back to torch elsewhere. The paged decode attention kernel supports decode
length 1, CUDA `float16`, positive sequence lengths, GQA/MQA, paged K/V shaped
`(num_blocks, kv_heads, block_size, head_dim)`, and the benchmark shapes used by
Phase 6/7. Unsupported shapes still use the torch fallback unless the caller
requires TileLang, in which case they fail explicitly.

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
TileLang kernel compilation also requires a modern CUDA toolkit in `PATH`; the
remote runner puts `/usr/local/cuda/bin` before `/usr/bin` because this H100
host also has an older `/usr/bin/nvcc` that cannot compile `sm_90a`.

This slice reaches the first Phase 7 performance milestone when the H100
artifact shows the TileLang paged decode kernel beating the torch gather
reference for at least one documented shape. Simple operator kernels are
correctness and integration kernels first; they are benchmarked against torch
references but are not required to win every shape. NCU profiling remains a
follow-up profiling artifact.

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

