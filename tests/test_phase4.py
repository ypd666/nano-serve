from __future__ import annotations

from nano_serve.engine import Engine, EngineConfig
from nano_serve.engine.batch import BatchKind
from nano_serve.engine.request import RequestMetrics, RequestState, RequestStatus
from nano_serve.sampling.base import SamplingParams
from nano_serve.scheduler.base import ScheduleBudget
from nano_serve.scheduler.continuous import ContinuousScheduler
from nano_serve.scheduler.policies import SchedulerPolicy


def test_continuous_scheduler_admits_fcfs_to_capacity() -> None:
    waiting = [_state("a", [1]), _state("b", [2]), _state("c", [3])]
    running: list[RequestState] = []

    plan = ContinuousScheduler().schedule(
        waiting=waiting,
        running=running,
        kv_cache=None,
        budget=ScheduleBudget(max_num_seqs=2, max_num_batched_tokens=16),
    )

    assert [request.request_id for request in running] == ["a", "b"]
    assert [request.request_id for request in waiting] == ["c"]
    assert plan.kind == BatchKind.PREFILL
    assert plan.request_ids == ["a", "b"]
    assert plan.num_prefill_tokens == 2
    assert plan.metadata["admitted_count"] == 2


def test_continuous_scheduler_respects_full_context_token_budget() -> None:
    running = [
        _state("a", [1, 2, 3], [4]),
        _state("b", [5, 6, 7], [8]),
    ]

    plan = ContinuousScheduler().schedule(
        waiting=[],
        running=running,
        kv_cache=None,
        budget=ScheduleBudget(max_num_seqs=2, max_num_batched_tokens=4),
    )

    assert plan.request_ids == ["a"]
    assert plan.input_token_ids == [[1, 2, 3, 4]]


def test_continuous_scheduler_decode_first_policy() -> None:
    waiting = [_state("new", [9])]
    running = [_state("decode", [1], [2])]

    plan = ContinuousScheduler(SchedulerPolicy.DECODE_FIRST).schedule(
        waiting=waiting,
        running=running,
        kv_cache=None,
        budget=ScheduleBudget(max_num_seqs=2, max_num_batched_tokens=16),
    )

    assert plan.kind == BatchKind.MIXED
    assert plan.request_ids == ["decode", "new"]
    assert plan.num_decode_tokens == 1
    assert plan.num_prefill_tokens == 1


def test_engine_step_continuous_removes_finished_and_admits_new_request() -> None:
    engine = Engine(
        EngineConfig(
            model_path="unused",
            scheduler="continuous",
            scheduler_policy=SchedulerPolicy.FCFS,
            max_num_seqs=2,
        )
    )
    engine.model_runner = _FakeBatchRunner([[9], [7, 8, 8]])
    engine.add_request([1], SamplingParams(max_tokens=1, stop_token_ids=(9,)), "req-a")
    engine.add_request([2], SamplingParams(max_tokens=3, stop_token_ids=(9,)), "req-b")
    stream_events = []
    batch_events = []

    assert engine.step(stream_callback=stream_events.append, batch_callback=batch_events.append)

    assert [state.request_id for state in engine.finished] == ["req-a"]
    assert [state.request_id for state in engine.running] == ["req-b"]
    assert stream_events[-2].request_id == "req-a"
    assert batch_events[0].metadata["num_waiting_reqs"] == 0

    engine.model_runner.add_sequence("req-c", [5])
    engine.add_request([3], SamplingParams(max_tokens=1), "req-c")

    assert engine.step(stream_callback=stream_events.append, batch_callback=batch_events.append)

    assert [state.request_id for state in engine.running] == ["req-b"]
    assert [state.request_id for state in engine.finished] == ["req-a", "req-c"]
    assert batch_events[2].metadata["batch_kind"] == "MIXED"
    assert batch_events[2].metadata["num_prefill_tokens"] == 1
    assert batch_events[2].metadata["num_decode_tokens"] == 1

    assert engine.step(stream_callback=stream_events.append, batch_callback=batch_events.append)
    assert [state.request_id for state in engine.finished] == ["req-a", "req-c", "req-b"]
    assert engine.running == []
    assert engine.waiting == []


def _state(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int] | None = None,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=list(output_token_ids or []),
        sampling_params=SamplingParams(),
        metrics=RequestMetrics(arrival_time_ns=len(request_id)),
        status=RequestStatus.DECODE if output_token_ids else RequestStatus.WAITING,
    )


class _FakeConfig:
    pad_token_id = 0
    eos_token_id = 9


class _FakeBatchModel:
    config = _FakeConfig()


class _FakeBatchRunner:
    def __init__(self, sequences: list[list[int]]) -> None:
        self.model = _FakeBatchModel()
        self.sequences_by_prompt = {index + 1: sequence for index, sequence in enumerate(sequences)}
        self.sequences_by_request = {
            f"req-{chr(ord('a') + index)}": sequence
            for index, sequence in enumerate(sequences)
        }

    def add_sequence(self, request_id: str, sequence: list[int]) -> None:
        self.sequences_by_request[request_id] = sequence
        self.sequences_by_prompt[len(self.sequences_by_prompt) + 1] = sequence

    def next_token_logits_batch(
        self,
        token_ids_batch: list[list[int]],
        *,
        pad_token_id: int = 0,
    ):
        del pad_token_id
        import torch

        logits = torch.full((len(token_ids_batch), 16), -1000.0)
        for row, token_ids in enumerate(token_ids_batch):
            prompt_id = token_ids[0]
            sequence = self.sequences_by_prompt[prompt_id]
            generated = len(token_ids) - 1
            token_id = sequence[min(generated, len(sequence) - 1)]
            logits[row, token_id] = 1000.0
        return logits
