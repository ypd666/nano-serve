from __future__ import annotations

import json
from pathlib import Path

from nano_serve.benchmark.compare import compare_runs, render_compare_markdown
from nano_serve.benchmark.datasets import load_sharegpt_dataset
from nano_serve.benchmark.phase0 import Phase0SmokeConfig, run_phase0_smoke
from nano_serve.cli import main
from nano_serve.observability import Event, JSONLEventWriter, read_jsonl_events


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
