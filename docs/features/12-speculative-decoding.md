# Speculative Decoding

## Goal

Use draft/verify algorithms to reduce target model decode iterations.

## Why It Exists

Autoregressive decode runs the target model once per token. Speculative
decoding can accept multiple tokens per target verification step when the draft
is accurate enough.

## Dependencies

- Robust request state machine.
- KV append for multiple tokens.
- Streaming that can emit multiple accepted tokens per iteration.
- Metrics that separate draft overhead from target savings.

## Interfaces

- `SpeculativeDecoder`
- `DraftModel`
- `Verifier`
- acceptance accounting
- KV rollback/update helpers
- `NGramSpeculator`
- `SpeculativeDecodeConfig`
- `SpeculativeDecodeMetrics`
- `EngineConfig.spec_decode="draft_model" | "ngram"`

Phase 11 implements a deterministic reference path for greedy speculation. A
draft model proposes up to `gamma` tokens. The verifier compares draft tokens
against the target model's greedy continuation and accepts the longest prefix
that matches. If all proposed tokens are accepted, the verifier also emits one
bonus target token from the verification pass. If a proposed token is rejected,
accepted tokens before the rejection are emitted and the target token at the
rejection position is emitted instead.

The reference path records the KV update plan as token append counts rather than
mutating real model KV tensors. This keeps the algorithm testable before
batched paged-KV rollback is wired into the model runner. Later phases can map
the same `kv_tokens_appended` and `rollback_tokens` metrics onto real cache
operations.

Sampling mode is included as an explicit fallback in Phase 11. When
`SamplingParams` request stochastic sampling, the decoder records
`sampling_fallback=True` and records target-only accounting without speculation.
This avoids claiming distribution-correct speculative sampling before the
acceptance ratio correction is implemented.

## Metrics

- draft tokens proposed,
- accepted tokens,
- acceptance length,
- target calls per output token,
- draft overhead,
- TPOT and E2E change,
- hostile/friendly workload split.
- rejection count,
- bonus target tokens,
- KV tokens appended,
- rollback tokens,
- sampling fallback count.

## Tests

- greedy draft/verify correctness,
- rejection path correctness,
- KV update after accepted tokens,
- sampling distribution test,
- batched requests with different accepted lengths.
- n-gram proposal correctness,
- benchmark JSONL schema.

## Benchmarks

- friendly workload,
- hostile workload,
- gamma sweep,
- draft size sweep,
- batch size interaction.

Phase 11 adds a deterministic speculative benchmark:

```bash
python -m nano_serve.cli phase11-speculative \
  --gamma-values 1,2,4,8 \
  --output-tokens 256 \
  --batch-size 4
```

The benchmark runs friendly and hostile synthetic target streams. Friendly runs
use a draft stream that mostly matches the target, while hostile runs use a
shifted draft stream to force frequent rejections. Each case records target
decode steps, draft tokens proposed, accepted tokens, acceptance rate,
acceptance length, target calls per output token, rejection count, bonus tokens,
KV append/rollback counts, and estimated speedup against non-speculative greedy
decode. The JSONL log emits `speculative_iteration`,
`speculative_request_end`, and `speculative_case` events.

## Exit Criteria

- Benchmarks show where speculation helps and where it hurts.
- Output distribution rules are documented for greedy vs sampling modes.

## References

- Speculative Decoding.
- Speculative Sampling.
- Medusa.
- EAGLE.

