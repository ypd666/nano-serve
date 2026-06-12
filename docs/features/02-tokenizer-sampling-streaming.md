# Tokenizer, Sampling, and Streaming

## Goal

Add tokenizer integration, sampling parameters, and token streaming callbacks.

## Why It Exists

Serving behavior depends on tokenization, stop conditions, and when tokens are
emitted. TTFT and TPOT are impossible to measure correctly without streaming
timestamps.

## Dependencies

- Infrastructure.
- Torch forwarding.

## Interfaces

- `TokenizerAdapter`
- `SamplingParams`
- `Sampler`
- `StreamCallback`
- stop condition helpers

## Metrics

- tokenizer latency,
- first token timestamp,
- inter-token timestamps,
- sampling latency,
- output token count.

## Tests

- tokenizer round trip for fixed prompts,
- greedy sampler determinism,
- top-k/top-p probability masking,
- stop token behavior,
- streaming event order.

## Current Implementation Slice

The Phase 1 offline path includes a text-only tokenizer wrapper, greedy
sampling, temperature/top-k/top-p sampling, and a token stream callback on
`Engine.generate()`. Streaming callbacks are emitted after each token is sampled
and before stop-condition handling exits the decode loop.

This slice is still single-request and offline-only. HTTP streaming and
multi-request streaming order are left for the API and scheduler milestones.

## Benchmarks

- sampling overhead under small logits,
- streaming overhead under long decode,
- tokenizer batch latency.

## Exit Criteria

- Sampling and streaming are independent from model execution.
- Request metrics can compute TTFT, TPOT, and E2E.

## References

- NVIDIA GenAI-Perf metric definitions.

