"""Benchmark report rendering."""

from __future__ import annotations

from pathlib import Path


def render_markdown_report(summary: dict[str, object]) -> str:
    lines = [
        "# nano-serve Benchmark Report",
        "",
        f"- Run ID: `{summary.get('run_id', 'unknown')}`",
        f"- Workload: `{summary.get('workload', 'unknown')}`",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Samples loaded: `{summary.get('samples_loaded', 0)}`",
        f"- Samples skipped: `{summary.get('samples_skipped', 0)}`",
        f"- Dataset total rows: `{summary.get('dataset_total', 0)}`",
        f"- Model checked: `{summary.get('model_checked', False)}`",
        f"- Model loaded: `{summary.get('model_loaded', False)}`",
        f"- Device backend: `{summary.get('device_backend', 'unknown')}`",
        "",
        "## Artifacts",
        "",
    ]
    artifacts = summary.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, path in artifacts.items():
            lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, summary: dict[str, object]) -> None:
    path.write_text(render_markdown_report(summary), encoding="utf-8")
