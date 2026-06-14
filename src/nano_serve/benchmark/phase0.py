"""Phase 0 smoke runner."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nano_serve.assets import AssetConfig
from nano_serve.benchmark.datasets import load_sharegpt_dataset
from nano_serve.benchmark.profiler import nvtx_label, nvtx_range
from nano_serve.benchmark.report import write_markdown_report
from nano_serve.engine.config import BenchmarkConfig, EngineConfig
from nano_serve.observability import Event, JSONLEventWriter, platform_event
from nano_serve.platform import detect_platform


LOGGER = logging.getLogger("nano_serve.phase0")


@dataclass(frozen=True)
class Phase0SmokeConfig:
    output_dir: Path
    num_samples: int = 8
    load_model: bool = False
    workload: str = "phase0_smoke"
    enable_nvtx: bool = False


def run_phase0_smoke(config: Phase0SmokeConfig) -> dict[str, object]:
    asset_config = AssetConfig.from_env()
    run_id = _run_id()
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"

    platform_info = detect_platform()
    engine_config = EngineConfig(benchmark=BenchmarkConfig(enable_nvtx=config.enable_nvtx))
    run_config = {
        "run_id": run_id,
        "workload": config.workload,
        "num_samples": config.num_samples,
        "load_model": config.load_model,
        "enable_nvtx": config.enable_nvtx,
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "engine_config": engine_config.to_dict(),
        "asset_config": {
            "model_path": str(asset_config.model_path),
            "dataset_path": str(asset_config.dataset_path),
            "model_id": asset_config.model_id,
            "dataset_repo_id": asset_config.dataset_repo_id,
            "dataset_filename": asset_config.dataset_filename,
        },
        "platform": platform_info.to_dict(),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("starting phase0 smoke run_id=%s", run_id)

    model_checked = False
    model_loaded = False
    model_error: str | None = None
    with (
        JSONLEventWriter(events_path) as writer,
        nvtx_range(nvtx_label("phase0", "run"), enabled=config.enable_nvtx),
    ):
        writer.write(Event("run_start", fields={"run_id": run_id, "workload": config.workload}))
        writer.write(platform_event(platform_info))

        with nvtx_range(nvtx_label("phase0", "asset_check"), enabled=config.enable_nvtx):
            model_files = _check_model_assets(asset_config.model_path)
        model_checked = True
        writer.write(
            Event(
                "asset_check",
                fields={
                    "kind": "model",
                    "path": asset_config.model_path,
                    "files": model_files,
                },
            )
        )
        LOGGER.info("checked model assets path=%s files=%s", asset_config.model_path, len(model_files))

        writer.write(Event("dataset_load_start", fields={"path": asset_config.dataset_path}))
        with nvtx_range(nvtx_label("phase0", "dataset_load"), enabled=config.enable_nvtx):
            dataset = load_sharegpt_dataset(
                asset_config.dataset_path,
                max_samples=config.num_samples,
            )
        writer.write(
            Event(
                "dataset_load_end",
                fields={
                    "path": dataset.path,
                    "samples_loaded": len(dataset.samples),
                    "samples_skipped": dataset.skipped,
                    "dataset_total": dataset.total,
                },
            )
        )
        LOGGER.info(
            "loaded dataset samples=%s skipped=%s total=%s",
            len(dataset.samples),
            dataset.skipped,
            dataset.total,
        )
        for sample in dataset.samples:
            writer.write(
                Event(
                    "sample_loaded",
                    fields={
                        "sample_id": sample.sample_id,
                        "source_index": sample.source_index,
                        "prompt_chars": len(sample.prompt),
                        "reference_output_chars": len(sample.reference_output),
                    },
                )
            )

        if config.load_model:
            writer.write(Event("model_load_start", fields={"path": asset_config.model_path}))
            try:
                with nvtx_range(nvtx_label("phase0", "model_load"), enabled=config.enable_nvtx):
                    _load_qwen_model(asset_config.model_path)
                model_loaded = True
                writer.write(Event("model_load_end", fields={"loaded": True}))
                LOGGER.info("loaded model path=%s", asset_config.model_path)
            except Exception as exc:
                model_error = str(exc)
                writer.write(Event("model_load_error", fields={"error": model_error}))
                LOGGER.exception("model load failed")
                raise
        else:
            writer.write(Event("model_load_skipped", fields={"reason": "load_model_false"}))
            LOGGER.info("skipped heavy model load")

        summary = {
            "run_id": run_id,
            "workload": config.workload,
            "status": "ok",
            "run_dir": str(run_dir),
            "samples_loaded": len(dataset.samples),
            "samples_skipped": dataset.skipped,
            "dataset_total": dataset.total,
            "model_checked": model_checked,
            "model_loaded": model_loaded,
            "model_error": model_error,
            "device_backend": platform_info.device_backend,
            "platform": platform_info.to_dict(),
            "artifacts": {
                "run_config": str(run_config_path),
                "events": str(events_path),
                "summary": str(summary_path),
                "report": str(report_path),
            },
        }
        writer.write(Event("run_end", fields=summary))

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report_path, summary)
    LOGGER.info("wrote phase0 artifacts run_dir=%s", run_dir)
    return summary


def _check_model_assets(model_path: Path) -> list[str]:
    required = {"config.json", "tokenizer.json", "model.safetensors.index.json"}
    present = {path.name for path in model_path.iterdir() if path.is_file()}
    missing = sorted(required - present)
    if missing:
        raise FileNotFoundError(f"model path is missing required files: {missing}")
    shards = sorted(path.name for path in model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("model path has no safetensors shards")
    return sorted(present)


def _load_qwen_model(model_path: Path) -> None:
    try:
        from transformers import AutoConfig, AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Heavy model loading requires `pip install -e .[torch]`."
        ) from exc
    AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    AutoProcessor.from_pretrained(str(model_path))
    AutoModel.from_pretrained(str(model_path), torch_dtype="auto", trust_remote_code=True)


def _run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()
