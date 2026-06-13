from __future__ import annotations

import json
from pathlib import Path

from nano_serve.benchmark.phase8 import (
    Phase8ChunkedPrefillBenchmarkConfig,
    run_phase8_chunked_prefill_benchmark,
)
from nano_serve.engine import Engine, EngineConfig
from nano_serve.engine.batch import BatchKind
from nano_serve.engine.request import RequestMetrics, RequestState, RequestStatus
from nano_serve.model.torch_runner import DecodeOutput, PrefillOutput
from nano_serve.sampling.base import SamplingParams
from nano_serve.scheduler.base import ScheduleBudget
from nano_serve.scheduler.chunked_prefill import ChunkedPrefillScheduler


def test_chunked_prefill_scheduler_advances_prompt_chunks_after_decode() -> None:
    waiting = [_state("new", [10, 11, 12, 13, 14])]
    running = [_state("decode", [1], [2], status=RequestStatus.DECODE)]

    plan = ChunkedPrefillScheduler().schedule(
        waiting=waiting,
        running=running,
        kv_cache=None,
        budget=ScheduleBudget(
            max_num_seqs=2,
            max_num_batched_tokens=4,
            max_prefill_tokens=2,
        ),
    )

    assert plan.kind == BatchKind.MIXED
    assert plan.request_ids == ["decode", "new"]
    assert plan.num_decode_tokens == 1
    assert plan.num_prefill_tokens == 2
    assert plan.input_token_ids == [[1, 2], [10, 11]]
    assert plan.metadata["prefill_chunks"] == [
        {"request_id": "new", "start": 0, "end": 2, "num_tokens": 2}
    ]


def test_chunked_prefill_engine_emits_first_token_only_after_final_chunk() -> None:
    engine = Engine(
        EngineConfig(
            model_path="unused",
            scheduler="chunked_prefill",
            kv_cache="none",
            max_num_seqs=1,
            max_num_batched_tokens=2,
            max_prefill_chunk_tokens=2,
        )
    )
    engine.model_runner = _FakeChunkRunner([7, 8])
    stream_events = []
    batch_events = []

    output = engine.generate_chunked_prefill(
        [([1, 2, 3, 4, 5], SamplingParams(max_tokens=2, stop_token_ids=(9,)))],
        request_ids=["req"],
        stream_callback=stream_events.append,
        batch_callback=batch_events.append,
    )

    assert output == [[7, 8]]
    assert [event.token_index for event in stream_events] == [0, 1]
    assert [event.metadata["num_prefill_tokens"] for event in batch_events[::2]] == [2, 2, 1, 0]
    assert [call[:3] for call in engine.model_runner.prefill_calls] == [
        (0, 2, False),
        (2, 4, False),
        (4, 5, True),
    ]
    assert engine.finished[0].prefill_cursor == 5


def test_phase8_benchmark_emits_chunked_prefill_events(tmp_path: Path) -> None:
    summary = run_phase8_chunked_prefill_benchmark(
        Phase8ChunkedPrefillBenchmarkConfig(
            output_dir=tmp_path,
            chunk_sizes=(2, 4),
            long_prompt_tokens=8,
            decode_requests=2,
            decode_tokens_per_request=4,
            max_num_seqs=3,
            max_num_batched_tokens=4,
        )
    )

    assert summary["phase"] == "phase8"
    assert summary["status"] == "ok"
    assert len(summary["cases"]) == 3
    event_path = Path(summary["artifacts"]["events"])
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    names = {event["name"] for event in events}
    assert "chunked_prefill_iteration_start" in names
    assert "chunked_prefill_iteration_end" in names
    assert "chunked_prefill_case" in names
    first_case = summary["cases"][0]
    assert first_case["mixed_iterations"] > 0


def _state(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int] | None = None,
    *,
    status: RequestStatus = RequestStatus.WAITING,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=list(output_token_ids or []),
        sampling_params=SamplingParams(),
        metrics=RequestMetrics(arrival_time_ns=len(request_id)),
        status=status,
        prefill_cursor=len(prompt_token_ids) if output_token_ids else 0,
    )


class _FakeConfig:
    pad_token_id = 0
    eos_token_id = 9


class _FakeModel:
    config = _FakeConfig()


class _FakeChunkRunner:
    def __init__(self, sequence: list[int]) -> None:
        self.model = _FakeModel()
        self.sequence = sequence
        self.prefill_calls: list[tuple[int, int, bool, list[int]]] = []
        self.decode_calls: list[list[int]] = []

    def prefill_chunk(
        self,
        prompt_token_ids: list[int],
        *,
        start: int,
        end: int,
        request_id: str | None = None,
        max_decode_tokens: int = 0,
        final_chunk: bool = False,
    ) -> PrefillOutput:
        del request_id, max_decode_tokens
        self.prefill_calls.append((start, end, final_chunk, list(prompt_token_ids[start:end])))
        return PrefillOutput(
            logits=_logits(self.sequence[0]) if final_chunk else None,
            metadata={
                "chunk_start": start,
                "chunk_end": end,
                "chunk_tokens": end - start,
                "kv_cache": "none",
            },
        )

    def decode(
        self,
        context_token_ids: list[int],
        *,
        new_token_id: int | None = None,
        request_id: str | None = None,
    ) -> DecodeOutput:
        del new_token_id, request_id
        self.decode_calls.append(list(context_token_ids))
        generated = len(context_token_ids) - 5
        return DecodeOutput(
            logits=_logits(self.sequence[min(generated, len(self.sequence) - 1)]),
            metadata={"kv_cache": "none"},
        )

    def free(self, request_id: str) -> None:
        del request_id


def _logits(token_id: int):
    import torch

    logits = torch.full((1, 16), -1000.0)
    logits[0, token_id] = 1000.0
    return logits
