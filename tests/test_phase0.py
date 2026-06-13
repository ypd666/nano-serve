from __future__ import annotations

import json
from pathlib import Path

from nano_serve.benchmark.compare import compare_runs, render_compare_markdown
from nano_serve.benchmark.datasets import load_sharegpt_dataset
from nano_serve.benchmark.offline import OfflineBenchmarkConfig, run_offline_benchmark
from nano_serve.benchmark.phase5 import Phase5KVBenchmarkConfig, run_phase5_kv_benchmark
from nano_serve.benchmark.phase0 import Phase0SmokeConfig, run_phase0_smoke
from nano_serve.engine.core import BatchEvent
from nano_serve.cli import main
from nano_serve.observability import Event, JSONLEventWriter, read_jsonl_events
from nano_serve.scheduler.policies import SchedulerPolicy


def test_jsonl_event_writer_roundtrip(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"

    with JSONLEventWriter(events_path) as writer:
        writer.write(Event("sample", fields={"path": tmp_path / "asset"}))

    events = read_jsonl_events(events_path)

    assert len(events) == 1
    assert events[0]["name"] == "sample"
    assert events[0]["fields"]["path"] == str(tmp_path / "asset")


def test_sharegpt_loader_fixture(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "sample-a",
                    "conversations": [
                        {"from": "human", "value": " hello "},
                        {"from": "gpt", "value": " world "},
                    ],
                },
                {"id": "bad", "conversations": [{"from": "human", "value": "only"}]},
                {
                    "conversations": [
                        {"from": "user", "value": "prompt"},
                        {"from": "assistant", "value": "answer"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    result = load_sharegpt_dataset(dataset_path, max_samples=2)

    assert result.total == 3
    assert result.skipped == 1
    assert [sample.sample_id for sample in result.samples] == ["sample-a", "2"]
    assert result.samples[0].prompt == "hello"
    assert result.samples[0].reference_output == "world"


def test_phase0_smoke_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    model_path = _write_fake_model(tmp_path / "model")
    dataset_path = _write_fake_dataset(tmp_path / "sharegpt.json")
    output_dir = tmp_path / "runs"
    monkeypatch.setenv("NANO_SERVE_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NANO_SERVE_DATASET_PATH", str(dataset_path))

    summary = run_phase0_smoke(
        Phase0SmokeConfig(output_dir=output_dir, num_samples=2, load_model=False)
    )

    assert summary["status"] == "ok"
    assert summary["samples_loaded"] == 2
    assert summary["model_checked"] is True
    assert summary["model_loaded"] is False

    artifacts = summary["artifacts"]
    assert isinstance(artifacts, dict)
    for path in artifacts.values():
        assert Path(path).exists()

    events = read_jsonl_events(Path(artifacts["events"]))
    assert [event["name"] for event in events] == [
        "run_start",
        "platform_detected",
        "asset_check",
        "dataset_load_start",
        "dataset_load_end",
        "sample_loaded",
        "sample_loaded",
        "model_load_skipped",
        "run_end",
    ]


def test_phase1_offline_benchmark_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    model_path = _write_fake_model(tmp_path / "model")
    dataset_path = _write_fake_dataset(tmp_path / "sharegpt.json")
    output_dir = tmp_path / "phase1-runs"
    monkeypatch.setenv("NANO_SERVE_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NANO_SERVE_DATASET_PATH", str(dataset_path))
    monkeypatch.setattr("nano_serve.benchmark.offline.TokenizerWrapper", _FakeTokenizer)
    monkeypatch.setattr("nano_serve.benchmark.offline.Engine", _FakeEngine)

    summary = run_offline_benchmark(
        OfflineBenchmarkConfig(
            output_dir=output_dir,
            num_samples=2,
            max_new_tokens=3,
            max_prompt_tokens=4,
            kv_cache="contiguous",
        )
    )

    assert summary["status"] == "ok"
    assert summary["phase"] == "phase1"
    assert summary["kv_cache"] == "contiguous"
    assert summary["samples_loaded"] == 2
    assert summary["total_output_tokens"] == 4
    assert summary["output_tokens_per_sec"] is not None
    assert summary["max_kv_bytes_used"] == 0

    artifacts = summary["artifacts"]
    assert isinstance(artifacts, dict)
    for path in artifacts.values():
        assert Path(path).exists()

    events = read_jsonl_events(Path(artifacts["events"]))
    assert [event["name"] for event in events] == [
        "run_start",
        "platform_detected",
        "dataset_load_end",
        "prefill_start",
        "prefill_end",
        "stream_token",
        "decode_step_start",
        "decode_step_end",
        "stream_token",
        "request_end",
        "prefill_start",
        "prefill_end",
        "stream_token",
        "decode_step_start",
        "decode_step_end",
        "stream_token",
        "request_end",
        "run_end",
    ]
    assert events[4]["fields"]["kv_cache"] == "none"
    assert events[7]["fields"]["kv_sequence_length"] == 0


def test_static_batch_offline_benchmark_writes_batch_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = _write_fake_model(tmp_path / "model")
    dataset_path = _write_fake_dataset(tmp_path / "sharegpt.json")
    output_dir = tmp_path / "phase3-runs"
    monkeypatch.setenv("NANO_SERVE_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NANO_SERVE_DATASET_PATH", str(dataset_path))
    monkeypatch.setattr("nano_serve.benchmark.offline.TokenizerWrapper", _FakeTokenizer)
    monkeypatch.setattr("nano_serve.benchmark.offline.Engine", _FakeStaticBatchEngine)

    summary = run_offline_benchmark(
        OfflineBenchmarkConfig(
            output_dir=output_dir,
            num_samples=2,
            max_new_tokens=2,
            max_prompt_tokens=4,
            scheduler="static_batch",
            batch_size=2,
        )
    )

    assert summary["status"] == "ok"
    assert summary["phase"] == "phase3"
    assert summary["scheduler"] == "static_batch"
    assert summary["batch_count"] == 1
    assert summary["max_batch_size"] == 2
    assert summary["total_padded_tokens"] == 2
    assert summary["total_inactive_slot_steps"] == 1

    events = read_jsonl_events(Path(summary["artifacts"]["events"]))
    assert [event["name"] for event in events] == [
        "run_start",
        "platform_detected",
        "dataset_load_end",
        "batch_prefill_start",
        "batch_prefill_end",
        "stream_token",
        "stream_token",
        "batch_decode_step_start",
        "batch_decode_step_end",
        "stream_token",
        "batch_end",
        "batch_request_end",
        "batch_request_end",
        "run_end",
    ]
    assert events[3]["fields"]["padded_tokens"] == 1
    assert events[7]["fields"]["inactive_slots"] == 1


def test_continuous_offline_benchmark_writes_iteration_events(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = _write_fake_model(tmp_path / "model")
    dataset_path = _write_fake_dataset(tmp_path / "sharegpt.json")
    output_dir = tmp_path / "phase4-runs"
    monkeypatch.setenv("NANO_SERVE_MODEL_PATH", str(model_path))
    monkeypatch.setenv("NANO_SERVE_DATASET_PATH", str(dataset_path))
    monkeypatch.setattr("nano_serve.benchmark.offline.TokenizerWrapper", _FakeTokenizer)
    monkeypatch.setattr("nano_serve.benchmark.offline.Engine", _FakeContinuousEngine)

    summary = run_offline_benchmark(
        OfflineBenchmarkConfig(
            output_dir=output_dir,
            num_samples=2,
            max_new_tokens=2,
            max_prompt_tokens=4,
            scheduler="continuous",
            scheduler_policy=SchedulerPolicy.DECODE_FIRST,
            batch_size=2,
            max_num_batched_tokens=16,
        )
    )

    assert summary["status"] == "ok"
    assert summary["phase"] == "phase4"
    assert summary["scheduler"] == "continuous"
    assert summary["scheduler_policy"] == "decode_first"
    assert summary["batch_count"] == 1
    assert summary["max_running_reqs"] == 2
    assert summary["max_waiting_reqs"] == 1
    assert summary["total_cpu_schedule_time_ms"] == 0.5

    events = read_jsonl_events(Path(summary["artifacts"]["events"]))
    assert [event["name"] for event in events] == [
        "run_start",
        "platform_detected",
        "dataset_load_end",
        "continuous_iteration_start",
        "stream_token",
        "stream_token",
        "continuous_iteration_end",
        "continuous_request_end",
        "continuous_request_end",
        "run_end",
    ]
    assert events[3]["fields"]["batch_kind"] == "PREFILL"
    assert events[3]["fields"]["num_waiting_reqs"] == 1


def test_phase5_kv_benchmark_writes_allocator_events(tmp_path: Path) -> None:
    summary = run_phase5_kv_benchmark(
        Phase5KVBenchmarkConfig(
            output_dir=tmp_path / "phase5-runs",
            num_blocks=4,
            block_size=2,
            num_requests=3,
            max_prefill_tokens=3,
            max_decode_tokens=2,
            seed=0,
        )
    )

    assert summary["status"] == "ok"
    assert summary["phase"] == "phase5"
    assert summary["num_blocks"] == 4
    assert "internal_fragmentation" in summary["peak"]
    assert "oom_count" in summary["final"]

    events = read_jsonl_events(Path(summary["artifacts"]["events"]))
    assert events[0]["name"] == "run_start"
    assert any(event["name"] == "paged_kv_prefill" for event in events)
    assert any(event["name"] == "paged_kv_free" for event in events)
    assert events[-1]["name"] == "run_end"


def test_compare_runs(tmp_path: Path) -> None:
    base = _write_summary(tmp_path / "base", "base", samples_loaded=2)
    candidate = _write_summary(tmp_path / "candidate", "candidate", samples_loaded=5)

    comparison = compare_runs(base, candidate)
    rendered = render_compare_markdown(comparison)

    assert comparison["samples_loaded_delta"] == 3
    assert comparison["status_changed"] is False
    assert "candidate" in rendered


def test_cli_phase0_assets_env(capsys) -> None:
    assert main(["assets", "env"]) == 0

    output = capsys.readouterr().out
    assert "NANO_SERVE_MODEL_PATH" in output
    assert "NANO_SERVE_DATASET_PATH" in output


def _write_fake_model(path: Path) -> Path:
    path.mkdir()
    for name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "model.safetensors-00001-of-00001.safetensors",
    ):
        (path / name).write_text("{}", encoding="utf-8")
    return path


def _write_fake_dataset(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "conversations": [
                        {"from": "human", "value": "prompt a"},
                        {"from": "gpt", "value": "answer a"},
                    ],
                },
                {
                    "id": "b",
                    "conversations": [
                        {"from": "human", "value": "prompt b"},
                        {"from": "gpt", "value": "answer b"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_summary(path: Path, run_id: str, *, samples_loaded: int) -> Path:
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "ok",
                "samples_loaded": samples_loaded,
                "samples_skipped": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


class _FakeTokenizer:
    eos_token_id = 15

    @classmethod
    def from_pretrained(cls, model_path: Path) -> "_FakeTokenizer":
        assert model_path.exists()
        return cls()

    def encode(self, text: str) -> list[int]:
        return [ord(char) % 16 for char in text]


class _FakeMetrics:
    ttft_ms = 1.0
    e2e_ms = 3.0

    def tpot_ms(self, output_tokens: int) -> float | None:
        return 1.0 if output_tokens > 1 else None


class _FakeState:
    stop_reason = "max_tokens"
    metrics = _FakeMetrics()


class _FakeEngine:
    def __init__(self, config: object) -> None:
        del config
        self.finished: list[_FakeState] = []

    def generate(
        self,
        prompt_token_ids: list[int],
        params: object,
        stream_callback: object = None,
        phase_callback: object = None,
    ) -> list[int]:
        del prompt_token_ids, params
        if callable(phase_callback):
            phase_callback(_FakePhaseEvent(phase="prefill", event="start", token_index=None))
            phase_callback(
                _FakePhaseEvent(
                    phase="prefill",
                    event="end",
                    token_index=None,
                    metadata={"kv_cache": "none", "kv_sequence_length": 0},
                )
            )
        if callable(stream_callback):
            stream_callback(_FakeStreamEvent(token_id=3, token_index=0))
        if callable(phase_callback):
            phase_callback(_FakePhaseEvent(phase="decode", event="start", token_index=1))
            phase_callback(
                _FakePhaseEvent(
                    phase="decode",
                    event="end",
                    token_index=1,
                    metadata={"kv_cache": "none", "kv_sequence_length": 0},
                )
            )
        if callable(stream_callback):
            stream_callback(_FakeStreamEvent(token_id=4, token_index=1))
        self.finished.append(_FakeState())
        return [3, 4]


class _FakeStaticBatchEngine:
    def __init__(self, config: object) -> None:
        del config
        self.finished: list[_FakeStaticState] = []

    def generate_static_batch(
        self,
        requests: list[tuple[list[int], object]],
        *,
        request_ids: list[str],
        stream_callback: object = None,
        batch_callback: object = None,
    ) -> list[list[int]]:
        del requests
        if callable(batch_callback):
            batch_callback(_fake_batch_event("prefill_start", 0, active=2, inactive=0, padded=1))
            batch_callback(_fake_batch_event("prefill_end", 0, active=2, inactive=0, padded=1))
        if callable(stream_callback):
            stream_callback(_FakeStreamEvent(request_id=request_ids[0], token_id=15, token_index=0))
            stream_callback(_FakeStreamEvent(request_id=request_ids[1], token_id=3, token_index=0))
        if callable(batch_callback):
            batch_callback(
                _fake_batch_event("decode_step_start", 1, active=1, inactive=1, padded=1)
            )
            batch_callback(
                _fake_batch_event("decode_step_end", 1, active=1, inactive=1, padded=1)
            )
        if callable(stream_callback):
            stream_callback(_FakeStreamEvent(request_id=request_ids[1], token_id=4, token_index=1))
        if callable(batch_callback):
            batch_callback(_fake_batch_event("batch_end", 2, active=0, inactive=2, padded=0))
        self.finished.extend(
            [
                _FakeStaticState(request_id=request_ids[0], stop_reason="eos_token"),
                _FakeStaticState(request_id=request_ids[1], stop_reason="max_tokens"),
            ]
        )
        return [[15], [3, 4]]


class _FakeContinuousEngine:
    def __init__(self, config: object) -> None:
        del config
        self.finished: list[_FakeStaticState] = []

    def generate_continuous(
        self,
        requests: list[tuple[list[int], object]],
        *,
        request_ids: list[str],
        stream_callback: object = None,
        batch_callback: object = None,
    ) -> list[list[int]]:
        del requests
        if callable(batch_callback):
            batch_callback(
                BatchEvent(
                    event="iteration_start",
                    iteration=0,
                    timestamp_ns=0,
                    metadata={
                        "batch_kind": "PREFILL",
                        "batch_size": 2,
                        "request_ids": request_ids,
                        "num_prefill_tokens": 8,
                        "num_decode_tokens": 0,
                        "num_running_reqs": 2,
                        "num_waiting_reqs": 1,
                        "real_tokens": 8,
                        "padded_tokens": 0,
                        "max_tokens_per_slot": 4,
                        "cpu_schedule_time_ms": 0.5,
                    },
                )
            )
        if callable(stream_callback):
            stream_callback(_FakeStreamEvent(request_id=request_ids[0], token_id=3, token_index=0))
            stream_callback(_FakeStreamEvent(request_id=request_ids[1], token_id=4, token_index=0))
        if callable(batch_callback):
            batch_callback(
                BatchEvent(
                    event="iteration_end",
                    iteration=0,
                    timestamp_ns=0,
                    metadata={
                        "batch_kind": "PREFILL",
                        "batch_size": 2,
                        "request_ids": request_ids,
                        "num_prefill_tokens": 8,
                        "num_decode_tokens": 0,
                        "num_running_reqs": 2,
                        "num_waiting_reqs": 1,
                        "real_tokens": 8,
                        "padded_tokens": 0,
                        "max_tokens_per_slot": 4,
                        "cpu_schedule_time_ms": 0.5,
                    },
                )
            )
        self.finished.extend(
            [
                _FakeStaticState(request_id=request_ids[0], stop_reason="max_tokens"),
                _FakeStaticState(request_id=request_ids[1], stop_reason="max_tokens"),
            ]
        )
        return [[3], [4]]


def _fake_batch_event(
    event: str,
    iteration: int,
    *,
    active: int,
    inactive: int,
    padded: int,
) -> BatchEvent:
    return BatchEvent(
        event=event,
        iteration=iteration,
        timestamp_ns=0,
        metadata={
            "batch_size": 2,
            "active_slots": active,
            "inactive_slots": inactive,
            "max_tokens_per_slot": 4,
            "real_tokens": 7,
            "padded_tokens": padded,
        },
    )


class _FakeStreamEvent:
    request_id = "fake-request"

    def __init__(
        self,
        *,
        token_id: int,
        token_index: int,
        request_id: str = "fake-request",
    ) -> None:
        self.request_id = request_id
        self.token_id = token_id
        self.token_index = token_index
        self.timestamp_ns = 0


class _FakePhaseEvent:
    request_id = "fake-request"
    num_tokens = 1

    def __init__(
        self,
        *,
        phase: str,
        event: str,
        token_index: int | None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.phase = phase
        self.event = event
        self.token_index = token_index
        self.timestamp_ns = 0
        self.metadata = metadata


class _FakeStaticState:
    metrics = _FakeMetrics()

    def __init__(self, *, request_id: str, stop_reason: str) -> None:
        self.request_id = request_id
        self.stop_reason = stop_reason
