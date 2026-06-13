# nano-serve TODO

This file is the implementation checklist. `README.md` keeps the human-facing
roadmap; this file is the more operational backlog.

## Milestone Rules

- Do not mark a feature done until it has a config flag, tests, benchmarks, and
  a design doc.
- Every benchmark run should emit JSONL events and a reproducible run config.
- Every optimization must have an ablation against the previous baseline.
- If a feature changes request state, KV ownership, scheduling, or metrics, also
  update `docs/architecture.md` and `docs/benchmarking.md`.

## M0: Infrastructure

- [x] Define `EngineConfig`.
- [x] Define `FeatureFlags`.
- [x] Define request and iteration metric schemas.
- [x] Implement JSONL event writer.
- [x] Implement benchmark workload registry.
- [x] Implement report generator.
- [x] Implement report comparison.
- [x] Add NVTX helper.
- [x] Add import smoke tests.
- [x] Add Qwen3.5-4B and ShareGPT asset downloader.
- [x] Add ShareGPT dataset fixture loading.
- [x] Add Phase 0 local smoke CLI and artifact generation.
- [x] Add macOS CPU-only and Linux NVIDIA CUDA platform policy.

## M1: Torch Single Request

- [x] Load HF config.
- [x] Load `safetensors`.
- [x] Implement minimal Llama/Qwen-style block.
- [x] Implement full-context forward.
- [x] Implement greedy decode.
- [x] Implement sampling params.
- [x] Add Hugging Face correctness oracle interface.
- [x] Compare logits with Hugging Face.
- [x] Measure single-request latency.

## M2: KV Cache

- [x] Split prefill and decode.
- [x] Implement contiguous KV layout.
- [x] Track per-request sequence length.
- [x] Validate RoPE position handling.
- [x] Compare cached decode with full forward.
- [x] Record KV memory usage.

## M3: Static Batching

- [x] Batch prefill with padding.
- [x] Batch decode.
- [x] Track inactive slots.
- [x] Track padding waste.
- [x] Benchmark equal-length and mixed-length prompts.

## M4: Continuous Batching

- [ ] Implement waiting/running/finished queues.
- [ ] Implement FCFS scheduler.
- [ ] Implement `Engine.step()`.
- [ ] Admit new requests during decode.
- [ ] Free finished requests immediately.
- [ ] Add batch timeline metric.
- [ ] Benchmark RPS sweep.
- [ ] Add vLLM and SGLang baseline benchmark scripts once local workloads are
      comparable.

## M5-M6: Paged KV and Reference Paged Attention

- [ ] Implement KV blocks.
- [ ] Implement block table.
- [ ] Implement free list allocator.
- [ ] Implement append/free/OOM behavior.
- [ ] Implement torch gather paged attention reference.
- [ ] Validate correctness against contiguous KV.
- [ ] Benchmark fragmentation and gather overhead.

## M7: TileLang Kernels

- [ ] Add TileLang dev dependency option.
- [ ] Implement RMSNorm kernel.
- [ ] Implement RoPE kernel.
- [ ] Implement SiLU-mul kernel.
- [ ] Implement paged decode attention kernel.
- [ ] Add NCU profile script.
- [ ] Compare with torch references.

## M8-M9: Chunked Prefill and Prefix Cache

- [ ] Add `prefill_cursor`.
- [ ] Add prefill chunk budget.
- [ ] Add mixed prefill/decode batch plan.
- [ ] Add block hash prefix cache.
- [ ] Add radix cache prototype.
- [ ] Add shared-prefix workload.
- [ ] Plot TTFT/TPOT frontier.

## M10-M12: Overlap, Spec Decode, Advanced Features

- [ ] Add tokenizer worker.
- [ ] Add scheduler/model double buffering.
- [ ] Add CUDA graph shape buckets.
- [ ] Add draft-model speculative decoding.
- [ ] Add n-gram speculation.
- [ ] Add quantization experiments.
- [ ] Add LoRA and structured output experiments.

## M13-M17: Distributed and Production-Like Serving

- [ ] Data-parallel replicas.
- [ ] Tensor parallelism.
- [ ] Pipeline parallelism.
- [ ] Expert parallelism.
- [ ] Multi-node worker launcher.
- [ ] PD disaggregation.
- [ ] AF disaggregation simulator and prototype.
- [ ] Prometheus metrics.
- [ ] Request tracing.
- [ ] Regression benchmark CI.
