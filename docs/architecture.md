# Architecture

`nano-serve` is organized around the engine loop, request state, scheduler, KV
cache manager, model runner, attention backend, sampler, and benchmark event
stream.

## Core Data Flow

```text
API/offline caller
  -> Engine.add_request()
  -> waiting queue
  -> Scheduler.schedule()
  -> BatchPlan
  -> ModelRunner.execute()
  -> AttentionBackend / KVCacheManager
  -> Sampler
  -> RequestState update
  -> stream token + metrics event
```

## Core Abstractions

- `RequestState`: one user request, including prompt tokens, generated tokens,
  status, metrics, prefill cursor, and KV handle.
- `BatchPlan`: one engine iteration's model input plan. It may contain prefill
  tokens, decode tokens, or a mixed chunked-prefill/decode plan.
- `Scheduler`: decides which waiting/running requests run in the next
  iteration under token, sequence, and KV budgets.
- `KVCacheManager`: owns KV memory, block tables, ref counts, prefix sharing,
  and eviction.
- `AttentionBackend`: computes attention against contiguous or paged KV.
- `ModelRunner`: executes one `BatchPlan` and returns logits/model output.
- `Sampler`: converts logits into token ids under sampling parameters.

## State Ownership

The engine owns request lifecycle. The scheduler may select requests and request
capacity checks, but it must not mutate KV internals. The KV cache manager owns
allocation, release, block tables, and prefix sharing. The model runner should
not hide request admission, stop conditions, or streaming behavior.

## Early Execution Modes

1. `single`: one request, full-context PyTorch forward.
2. `kv_single`: one request, prefill/decode split with contiguous KV.
3. `static_batch`: fixed batch with padding and inactive slots.
4. `continuous`: iteration-level scheduling with dynamic admission/removal.
5. `chunked_prefill`: mixed prefill/decode scheduling for long prompts.

## Phase 2 KV Ownership

`EngineConfig.kv_cache` selects the single-request execution path:

- `none`: Phase 1 full-context baseline. Decode reruns prompt plus generated
  tokens.
- `contiguous`: Phase 2 cached decode. Prefill allocates one contiguous
  per-request cache, and each decode step consumes only the newest token plus
  cached state.

The engine owns request lifecycle and calls `ModelRunner.free(request_id)` when
the request finishes. The model runner owns the `ContiguousKVCache` instance for
the single-request path. Request state records the current block table and phase
metadata, but does not mutate cache tensors directly.

For Qwen3.5-4B, contiguous cache contains two state types:

- full-attention layers store per-layer K/V tensors,
- linear-attention layers store recurrent state plus causal convolution window
  state.

## Distributed Boundary

Distributed features should reuse the same high-level engine contract:

- DP duplicates the engine/model workers behind a router.
- TP/PP/EP change the model runner and worker topology.
- PD splits prefill and decode workers, requiring explicit KV transfer.
- AF splits Attention and FFN/MoE, requiring per-step activation transfer.

Do not start distributed implementation until single-node state, metrics, and
KV ownership are stable.

