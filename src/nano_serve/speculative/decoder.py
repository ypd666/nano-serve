"""Greedy speculative decoding reference path."""

from __future__ import annotations

from dataclasses import dataclass, field

from nano_serve.sampling.base import SamplingParams
from nano_serve.speculative.draft_model import DraftModel
from nano_serve.speculative.verifier import VerificationResult, Verifier


@dataclass(frozen=True)
class SpeculativeDecodeConfig:
    gamma: int = 4
    max_tokens: int = 16

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class SpeculativeDecodeMetrics:
    draft_tokens_proposed: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    target_calls: int = 0
    iterations: int = 0
    rejection_count: int = 0
    bonus_tokens: int = 0
    rollback_tokens: int = 0
    kv_tokens_appended: int = 0
    sampling_fallback: bool = False
    acceptance_lengths: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.accepted_tokens / self.draft_tokens_proposed

    @property
    def target_calls_per_output_token(self) -> float:
        if self.emitted_tokens == 0:
            return 0.0
        return self.target_calls / self.emitted_tokens

    @property
    def mean_acceptance_length(self) -> float:
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    def to_dict(self) -> dict[str, object]:
        return {
            "draft_tokens_proposed": self.draft_tokens_proposed,
            "accepted_tokens": self.accepted_tokens,
            "emitted_tokens": self.emitted_tokens,
            "target_calls": self.target_calls,
            "iterations": self.iterations,
            "rejection_count": self.rejection_count,
            "bonus_tokens": self.bonus_tokens,
            "rollback_tokens": self.rollback_tokens,
            "kv_tokens_appended": self.kv_tokens_appended,
            "sampling_fallback": self.sampling_fallback,
            "acceptance_rate": self.acceptance_rate,
            "target_calls_per_output_token": self.target_calls_per_output_token,
            "mean_acceptance_length": self.mean_acceptance_length,
            "acceptance_lengths": list(self.acceptance_lengths),
        }


@dataclass(frozen=True)
class SpeculativeDecodeResult:
    output_token_ids: list[int]
    metrics: SpeculativeDecodeMetrics
    iterations: list[VerificationResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_token_ids": list(self.output_token_ids),
            "metrics": self.metrics.to_dict(),
            "iterations": [iteration.to_dict() for iteration in self.iterations],
        }


class SpeculativeDecoder:
    def __init__(self, draft_model: DraftModel, verifier: Verifier) -> None:
        self.draft_model = draft_model
        self.verifier = verifier

    def decode(
        self,
        prompt_token_ids: list[int],
        *,
        config: SpeculativeDecodeConfig,
        sampling_params: SamplingParams | None = None,
    ) -> SpeculativeDecodeResult:
        params = sampling_params or SamplingParams(max_tokens=config.max_tokens)
        if not _is_greedy(params):
            return _sampling_fallback(prompt_token_ids, config=config)

        context = list(prompt_token_ids)
        output: list[int] = []
        metrics = SpeculativeDecodeMetrics()
        iterations: list[VerificationResult] = []
        while len(output) < config.max_tokens:
            remaining = config.max_tokens - len(output)
            draft = self.draft_model.propose(context, max_tokens=min(config.gamma, remaining))
            if not draft:
                break
            verification = self.verifier.verify(context, draft)
            emitted = verification.emitted_token_ids[:remaining]
            if not emitted:
                break
            output.extend(emitted)
            context.extend(emitted)
            iterations.append(verification)
            _accumulate(metrics, verification, emitted_count=len(emitted))
        return SpeculativeDecodeResult(
            output_token_ids=output,
            metrics=metrics,
            iterations=iterations,
        )


def decode_batch(
    decoders: list[SpeculativeDecoder],
    prompts: list[list[int]],
    *,
    config: SpeculativeDecodeConfig,
    sampling_params: SamplingParams | None = None,
) -> list[SpeculativeDecodeResult]:
    if len(decoders) != len(prompts):
        raise ValueError("decoders length must match prompts length")
    return [
        decoder.decode(prompt, config=config, sampling_params=sampling_params)
        for decoder, prompt in zip(decoders, prompts, strict=True)
    ]


def _accumulate(
    metrics: SpeculativeDecodeMetrics,
    verification: VerificationResult,
    *,
    emitted_count: int,
) -> None:
    metrics.draft_tokens_proposed += verification.draft_tokens_proposed
    metrics.accepted_tokens += verification.accepted_tokens
    metrics.emitted_tokens += emitted_count
    metrics.target_calls += verification.target_calls
    metrics.iterations += 1
    metrics.rejection_count += 1 if verification.rejected else 0
    metrics.bonus_tokens += 1 if verification.bonus_token_id is not None else 0
    metrics.rollback_tokens += verification.rollback_tokens
    metrics.kv_tokens_appended += emitted_count
    metrics.acceptance_lengths.append(verification.accepted_tokens)


def _sampling_fallback(
    prompt_token_ids: list[int],
    *,
    config: SpeculativeDecodeConfig,
) -> SpeculativeDecodeResult:
    del prompt_token_ids
    metrics = SpeculativeDecodeMetrics(
        emitted_tokens=config.max_tokens,
        target_calls=config.max_tokens,
        iterations=config.max_tokens,
        kv_tokens_appended=config.max_tokens,
        sampling_fallback=True,
    )
    return SpeculativeDecodeResult(
        output_token_ids=[],
        metrics=metrics,
        iterations=[],
    )


def _is_greedy(params: SamplingParams) -> bool:
    return params.temperature == 0 or (
        params.temperature == 1.0 and params.top_k is None and params.top_p is None
    )
