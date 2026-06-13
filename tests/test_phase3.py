from __future__ import annotations

from nano_serve.engine import Engine, EngineConfig
from nano_serve.engine.request import RequestMetrics, RequestState, RequestStatus
from nano_serve.model.torch_runner import TorchModelRunner
from nano_serve.sampling.base import SamplingParams
from nano_serve.scheduler.base import ScheduleBudget
from nano_serve.scheduler.static_batch import StaticBatchScheduler, static_waste


def test_torch_runner_next_token_logits_batch_right_pads_and_gathers() -> None:
    model = _PaddingAwareModel()
    runner = TorchModelRunner(model=model)

    logits = runner.next_token_logits_batch([[1, 2, 3], [4]], pad_token_id=0)

    assert model.last_input_ids.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert logits.argmax(dim=-1).tolist() == [3, 4]


def test_static_waste_counts_padding_and_inactive_slots() -> None:
    active = _state("active", [1, 2], [3], RequestStatus.DECODE)
    finished = _state("finished", [4], [], RequestStatus.FINISHED)

    waste = static_waste([active, finished])

    assert waste.batch_size == 2
    assert waste.active_slots == 1
    assert waste.inactive_slots == 1
    assert waste.max_tokens_per_slot == 3
    assert waste.real_tokens == 4
    assert waste.padded_tokens == 2


def test_static_batch_scheduler_selects_fixed_waiting_group() -> None:
    requests = [
        _state("a", [1, 2], [], RequestStatus.WAITING),
        _state("b", [3], [], RequestStatus.WAITING),
        _state("c", [4], [], RequestStatus.WAITING),
    ]

    plan = StaticBatchScheduler().schedule(
        waiting=requests,
        running=[],
        kv_cache=None,
        budget=ScheduleBudget(max_num_seqs=2, max_num_batched_tokens=16),
    )

    assert plan.request_ids == ["a", "b"]
    assert plan.input_token_ids == [[1, 2], [3]]
    assert plan.metadata["padded_tokens"] == 1


def test_static_batch_scheduler_decode_uses_full_contexts() -> None:
    requests = [
        _state("a", [1, 2], [5], RequestStatus.DECODE),
        _state("b", [3], [], RequestStatus.FINISHED),
    ]

    plan = StaticBatchScheduler().schedule(
        waiting=[],
        running=requests,
        kv_cache=None,
        budget=ScheduleBudget(max_num_seqs=2, max_num_batched_tokens=16),
    )

    assert plan.input_token_ids == [[1, 2, 5], [3]]
    assert plan.num_prefill_tokens == 0
    assert plan.num_decode_tokens == 1
    assert plan.metadata["inactive_slots"] == 1


def test_engine_static_batch_keeps_slots_and_honors_per_request_stop() -> None:
    engine = Engine(EngineConfig(model_path="unused", scheduler="static_batch", max_num_seqs=2))
    engine.model_runner = _FakeBatchRunner([[9, 9, 9], [7, 8, 8]])
    stream_events = []
    batch_events = []

    output_token_ids = engine.generate_static_batch(
        [
            ([1, 2], SamplingParams(max_tokens=3, stop_token_ids=(9,))),
            ([3], SamplingParams(max_tokens=3, stop_token_ids=(9,))),
        ],
        request_ids=["req-a", "req-b"],
        stream_callback=stream_events.append,
        batch_callback=batch_events.append,
    )

    assert output_token_ids == [[9], [7, 8, 8]]
    assert [(event.request_id, event.token_id, event.token_index) for event in stream_events] == [
        ("req-a", 9, 0),
        ("req-b", 7, 0),
        ("req-b", 8, 1),
        ("req-b", 8, 2),
    ]
    assert [event.event for event in batch_events] == [
        "prefill_start",
        "prefill_end",
        "decode_step_start",
        "decode_step_end",
        "decode_step_start",
        "decode_step_end",
        "batch_end",
    ]
    assert batch_events[2].metadata["inactive_slots"] == 1
    assert batch_events[2].metadata["padded_tokens"] == 1
    assert [state.request_id for state in engine.finished] == ["req-a", "req-b"]
    assert [state.stop_reason for state in engine.finished] == ["eos_token", "max_tokens"]
    assert engine.model_runner.calls == [
        [[1, 2], [3]],
        [[1, 2, 9], [3, 7]],
        [[1, 2, 9], [3, 7, 8]],
    ]


def _state(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    status: RequestStatus,
) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        sampling_params=SamplingParams(),
        metrics=RequestMetrics(arrival_time_ns=0),
        status=status,
    )


class _PaddingAwareModel:
    device = "cpu"

    def __call__(self, input_ids):
        import torch

        self.last_input_ids = input_ids.detach().clone()
        logits = torch.full((*input_ids.shape, 16), -1000.0)
        logits.scatter_(2, input_ids.unsqueeze(-1), 1000.0)
        return logits


class _FakeConfig:
    pad_token_id = 0
    eos_token_id = 9


class _FakeBatchModel:
    config = _FakeConfig()


class _FakeBatchRunner:
    def __init__(self, sequences: list[list[int]]) -> None:
        self.model = _FakeBatchModel()
        self.sequences = sequences
        self.prompt_lengths: list[int] | None = None
        self.calls: list[list[list[int]]] = []

    def next_token_logits_batch(
        self,
        token_ids_batch: list[list[int]],
        *,
        pad_token_id: int = 0,
    ):
        del pad_token_id
        import torch

        if self.prompt_lengths is None:
            self.prompt_lengths = [len(token_ids) for token_ids in token_ids_batch]
        self.calls.append([list(token_ids) for token_ids in token_ids_batch])

        logits = torch.full((len(token_ids_batch), 16), -1000.0)
        for row, token_ids in enumerate(token_ids_batch):
            generated = len(token_ids) - self.prompt_lengths[row]
            token_id = self.sequences[row][min(generated, len(self.sequences[row]) - 1)]
            logits[row, token_id] = 1000.0
        return logits
