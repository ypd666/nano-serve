"""Benchmark comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_summary(path_or_run_dir: Path) -> dict[str, object]:
    path = path_or_run_dir / "summary.json" if path_or_run_dir.is_dir() else path_or_run_dir
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def compare_runs(base: Path, candidate: Path) -> dict[str, object]:
    base_summary = load_summary(base)
    candidate_summary = load_summary(candidate)
    return {
        "base_run_id": base_summary.get("run_id"),
        "candidate_run_id": candidate_summary.get("run_id"),
        "samples_loaded_delta": _delta(base_summary, candidate_summary, "samples_loaded"),
        "samples_skipped_delta": _delta(base_summary, candidate_summary, "samples_skipped"),
        "status_changed": base_summary.get("status") != candidate_summary.get("status"),
        "base_status": base_summary.get("status"),
        "candidate_status": candidate_summary.get("status"),
    }


def render_compare_markdown(comparison: dict[str, object]) -> str:
    return "\n".join(
        [
            "# nano-serve Benchmark Comparison",
            "",
            f"- Base run: `{comparison.get('base_run_id')}`",
            f"- Candidate run: `{comparison.get('candidate_run_id')}`",
            f"- Samples loaded delta: `{comparison.get('samples_loaded_delta')}`",
            f"- Samples skipped delta: `{comparison.get('samples_skipped_delta')}`",
            f"- Status changed: `{comparison.get('status_changed')}`",
            f"- Base status: `{comparison.get('base_status')}`",
            f"- Candidate status: `{comparison.get('candidate_status')}`",
            "",
        ]
    )


def _delta(base: dict[str, object], candidate: dict[str, object], key: str) -> int | None:
    base_value = base.get(key)
    candidate_value = candidate.get(key)
    if not isinstance(base_value, int) or not isinstance(candidate_value, int):
        return None
    return candidate_value - base_value
