from __future__ import annotations

import json
from pathlib import Path

from nano_serve.benchmark.phase11 import (
    Phase11SpeculativeBenchmarkConfig,
    run_phase11_speculative_benchmark,
)
from nano_serve.speculative import (
    GreedyTokenStreamVerifier,
    NGramSpeculator,
    SpeculativeDecodeConfig,
    SpeculativeDecoder,
    StaticDraftModel,
    decode_batch,
)


def test_speculative_decoder_accepts_all_draft_tokens_and_bonus() -> None:
    target = [10, 11, 12, 13]
    decoder = SpeculativeDecoder(
        StaticDraftModel(target, base_context_len=2),
        GreedyTokenStreamVerifier(target, base_context_len=2),
    )

    result = decoder.decode([1, 2], config=SpeculativeDecodeConfig(gamma=2, max_tokens=3))

    assert result.output_token_ids == [10, 11, 12]
    assert result.metrics.accepted_tokens == 2
    assert result.metrics.bonus_tokens == 1
    assert result.metrics.target_calls == 1
    assert result.metrics.kv_tokens_appended == 3


def test_speculative_decoder_replaces_rejected_token() -> None:
    decoder = SpeculativeDecoder(
        StaticDraftModel([10, 99, 12], base_context_len=1),
        GreedyTokenStreamVerifier([10, 11, 12], base_context_len=1),
    )

    result = decoder.decode([1], config=SpeculativeDecodeConfig(gamma=3, max_tokens=3))

    assert result.output_token_ids == [10, 11, 12]
    assert result.metrics.rejection_count == 1
    assert result.metrics.rollback_tokens == 2
    assert result.iterations[0].rejected_token_id == 99
    assert result.iterations[0].replacement_token_id == 11


def test_ngram_speculator_reuses_prior_suffix_continuation() -> None:
    speculator = NGramSpeculator(ngram_size=2)

    proposal = speculator.propose([1, 2, 3, 4, 2, 3], max_tokens=2)

    assert proposal == [4, 2]


def test_decode_batch_allows_different_acceptance_lengths() -> None:
    first = SpeculativeDecoder(
        StaticDraftModel([1, 2, 3], base_context_len=0),
        GreedyTokenStreamVerifier([1, 2, 3], base_context_len=0),
    )
    second = SpeculativeDecoder(
        StaticDraftModel([9, 8, 7], base_context_len=0),
        GreedyTokenStreamVerifier([9, 0, 7], base_context_len=0),
    )

    results = decode_batch(
        [first, second],
        [[], []],
        config=SpeculativeDecodeConfig(gamma=3, max_tokens=3),
    )

    assert [result.output_token_ids for result in results] == [[1, 2, 3], [9, 0, 7]]
    assert [result.metrics.mean_acceptance_length for result in results] == [3.0, 1.0]


def test_phase11_benchmark_emits_speculative_events(tmp_path: Path) -> None:
    summary = run_phase11_speculative_benchmark(
        Phase11SpeculativeBenchmarkConfig(
            output_dir=tmp_path,
            gamma_values=(1, 2),
            output_tokens=16,
            batch_size=2,
            prompt_tokens=4,
        )
    )

    assert summary["phase"] == "phase11"
    assert summary["status"] == "ok"
    assert len(summary["cases"]) == 4
    assert summary["best_speedup_case"]["estimated_speedup"] > 1
    events = [
        json.loads(line)
        for line in Path(summary["artifacts"]["events"]).read_text(encoding="utf-8").splitlines()
    ]
    names = {event["name"] for event in events}
    assert "speculative_iteration" in names
    assert "speculative_request_end" in names
    assert "speculative_case" in names
