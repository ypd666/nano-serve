from __future__ import annotations

import os
from pathlib import Path

import pytest

from nano_serve.assets import load_dotenv
from nano_serve.engine import Engine, EngineConfig
from nano_serve.engine.batch import BatchKind, BatchPlan
from nano_serve.model import HuggingFaceOracle, TokenizerWrapper
from nano_serve.model.loader import ModelLoader, ModelSpec
from nano_serve.model.torch_runner import TorchModelRunner
from nano_serve.sampling.base import SamplingParams


def test_model_loader_reads_qwen35_config() -> None:
    model_path = _model_path_or_skip()
    config = ModelLoader().load_config(ModelSpec(model_path=model_path))

    assert config["model_type"] == "qwen3_5"
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert config["text_config"]["num_hidden_layers"] == 32
    assert config["text_config"]["hidden_size"] == 2560


def test_tokenizer_wrapper_roundtrip() -> None:
    model_path = _model_path_or_skip()
    tokenizer = TokenizerWrapper.from_pretrained(model_path)

    token_ids = tokenizer.encode("hello qwen")
    decoded = tokenizer.decode(token_ids)

    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert "hello" in decoded.lower()
    assert tokenizer.eos_token_id is not None


def test_engine_generate_uses_full_context_greedy_runner() -> None:
    engine = Engine(EngineConfig(model_path="unused"))
    engine.model_runner = _FakeRunner([7, 8, 9])
    stream_events = []
    phase_events = []

    output_token_ids = engine.generate(
        [1, 2, 3],
        SamplingParams(max_tokens=3, stop_token_ids=(9,)),
        stream_events.append,
        phase_events.append,
    )

    assert output_token_ids == [7, 8, 9]
    assert [event.token_id for event in stream_events] == [7, 8, 9]
    assert [event.token_index for event in stream_events] == [0, 1, 2]
    assert [(event.phase, event.event, event.token_index) for event in phase_events] == [
        ("prefill", "start", None),
        ("prefill", "end", None),
        ("decode", "start", 1),
        ("decode", "end", 1),
        ("decode", "start", 2),
        ("decode", "end", 2),
    ]
    assert len(engine.finished) == 1
    state = engine.finished[0]
    assert state.output_token_ids == [7, 8, 9]
    assert state.stop_reason == "eos_token"
    assert state.metrics.ttft_ms is not None
    assert state.metrics.e2e_ms is not None
    assert engine.model_runner.calls == [
        ("prefill", [1, 2, 3]),
        ("decode", [1, 2, 3, 7]),
        ("decode", [1, 2, 3, 7, 8]),
    ]


def test_engine_generate_uses_top_k_sampler_when_requested() -> None:
    engine = Engine(EngineConfig(model_path="unused"))
    engine.model_runner = _FixedLogitsRunner([[1.0, 3.0, 2.0]])

    output_token_ids = engine.generate(
        [1, 2, 3],
        SamplingParams(max_tokens=1, temperature=1.0, top_k=1),
    )

    assert output_token_ids == [1]


@pytest.mark.skipif(
    os.environ.get("NANO_SERVE_RUN_HEAVY_TESTS") != "1",
    reason="set NANO_SERVE_RUN_HEAVY_TESTS=1 to load model weights",
)
def test_hf_oracle_next_token_logits_shape() -> None:
    model_path = _model_path_or_skip()
    oracle = HuggingFaceOracle.from_pretrained(model_path, dtype="bfloat16")
    token_ids = oracle.encode("hello qwen")

    logits = oracle.next_token_logits(token_ids)

    assert tuple(logits.shape) == (1, 248320)


@pytest.mark.skipif(
    os.environ.get("NANO_SERVE_RUN_HEAVY_TESTS") != "1",
    reason="set NANO_SERVE_RUN_HEAVY_TESTS=1 to load model weights",
)
def test_torch_model_loader_next_token_logits_shape() -> None:
    model_path = _model_path_or_skip()
    model = ModelLoader().load(ModelSpec(model_path=model_path, dtype="bfloat16"))
    tokenizer = TokenizerWrapper.from_pretrained(model_path)
    token_ids = tokenizer.encode("hello qwen")

    logits = model.next_token_logits(_input_ids(token_ids, model.device))

    assert tuple(logits.shape) == (1, 248320)


@pytest.mark.skipif(
    os.environ.get("NANO_SERVE_RUN_HEAVY_TESTS") != "1",
    reason="set NANO_SERVE_RUN_HEAVY_TESTS=1 to load model weights",
)
def test_torch_runner_single_request_logits_shape() -> None:
    model_path = _model_path_or_skip()
    runner = TorchModelRunner.from_model_spec(ModelSpec(model_path=model_path, dtype="bfloat16"))
    token_ids = TokenizerWrapper.from_pretrained(model_path).encode("hello qwen")
    batch = BatchPlan(
        kind=BatchKind.PREFILL,
        request_ids=["req-0"],
        input_token_ids=[token_ids],
        num_prefill_tokens=len(token_ids),
    )

    output = runner.execute(batch)

    assert tuple(output.logits.shape) == (1, len(token_ids), 248320)
    assert output.metadata["runner"] == "torch_full_context"


@pytest.mark.skipif(
    os.environ.get("NANO_SERVE_RUN_HEAVY_TESTS") != "1",
    reason="set NANO_SERVE_RUN_HEAVY_TESTS=1 to load model weights",
)
def test_torch_forwarding_matches_hf_oracle_for_short_prompt() -> None:
    model_path = _model_path_or_skip()
    token_ids = TokenizerWrapper.from_pretrained(model_path).encode("hello qwen phase one")

    oracle = HuggingFaceOracle.from_pretrained(model_path, dtype="bfloat16")
    expected = oracle.next_token_logits(token_ids).detach().float().cpu()
    del oracle
    _empty_cuda_cache()

    runner = TorchModelRunner.from_model_spec(ModelSpec(model_path=model_path, dtype="bfloat16"))
    actual = runner.next_token_logits(token_ids).detach().float().cpu()

    assert actual.shape == expected.shape
    _assert_close_logits(actual, expected)


@pytest.mark.skipif(
    os.environ.get("NANO_SERVE_RUN_HEAVY_TESTS") != "1",
    reason="set NANO_SERVE_RUN_HEAVY_TESTS=1 to load model weights",
)
def test_engine_generate_with_real_qwen35_smoke() -> None:
    model_path = _model_path_or_skip()
    tokenizer = TokenizerWrapper.from_pretrained(model_path)
    engine = Engine(EngineConfig(model_path=str(model_path)))
    token_ids = tokenizer.encode("hello qwen")

    output_token_ids = engine.generate(token_ids, SamplingParams(max_tokens=2))

    assert len(output_token_ids) == 2
    assert all(isinstance(token_id, int) for token_id in output_token_ids)


def _model_path_or_skip() -> Path:
    load_dotenv()
    value = os.environ.get("NANO_SERVE_MODEL_PATH")
    if not value:
        pytest.skip("NANO_SERVE_MODEL_PATH is not set")

    path = Path(value)
    if not path.exists():
        pytest.skip(f"NANO_SERVE_MODEL_PATH does not exist: {path}")
    return path


def _input_ids(token_ids: list[int], device: object):
    import torch

    return torch.tensor([token_ids], dtype=torch.long, device=device)


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _assert_close_logits(actual, expected) -> None:
    import torch

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-1)


class _FakeRunner:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.calls: list[tuple[str, list[int]]] = []

    def next_token_logits(self, token_ids: list[int]):
        import torch

        next_token_id = self.token_ids[len(self.calls) - 1]
        logits = torch.full((1, 16), -1000.0)
        logits[0, next_token_id] = 1000.0
        return logits

    def prefill(self, prompt_token_ids: list[int]):
        self.calls.append(("prefill", list(prompt_token_ids)))
        return _FakeRunnerOutput(self.next_token_logits(prompt_token_ids))

    def decode(self, context_token_ids: list[int], *, new_token_id: int | None = None):
        del new_token_id
        self.calls.append(("decode", list(context_token_ids)))
        return _FakeRunnerOutput(self.next_token_logits(context_token_ids))


class _FixedLogitsRunner:
    def __init__(self, logits: list[list[float]]) -> None:
        self.logits = logits
        self.index = 0

    def next_token_logits(self, token_ids: list[int]):
        del token_ids
        import torch

        logits = torch.tensor([self.logits[self.index]])
        self.index += 1
        return logits

    def prefill(self, prompt_token_ids: list[int]):
        return _FakeRunnerOutput(self.next_token_logits(prompt_token_ids))

    def decode(self, context_token_ids: list[int], *, new_token_id: int | None = None):
        del new_token_id
        return _FakeRunnerOutput(self.next_token_logits(context_token_ids))


class _FakeRunnerOutput:
    def __init__(self, logits: object) -> None:
        self.logits = logits
