"""Benchmark report rendering."""

from __future__ import annotations

from pathlib import Path


def render_markdown_report(summary: dict[str, object]) -> str:
    platform = summary.get("platform", {})
    device_backend = summary.get("device_backend")
    if device_backend is None and isinstance(platform, dict):
        device_backend = platform.get("device_backend", "unknown")

    lines = [
        "# nano-serve Benchmark Report",
        "",
        f"- Run ID: `{summary.get('run_id', 'unknown')}`",
        f"- Phase: `{summary.get('phase', 'unknown')}`",
        f"- Workload: `{summary.get('workload', 'unknown')}`",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Samples loaded: `{summary.get('samples_loaded', 0)}`",
        f"- Dataset total rows: `{summary.get('dataset_total', 0)}`",
        f"- Device backend: `{device_backend}`",
        "",
    ]
    if "samples_skipped" in summary:
        lines.append(f"- Samples skipped: `{summary.get('samples_skipped', 0)}`")
    if "model_checked" in summary:
        lines.append(f"- Model checked: `{summary.get('model_checked')}`")
    if "model_loaded" in summary:
        lines.append(f"- Model loaded: `{summary.get('model_loaded')}`")
    if "output_tokens_per_sec" in summary:
        lines.extend(
            [
                f"- Output tokens/s: `{summary.get('output_tokens_per_sec')}`",
                f"- Total tokens/s: `{summary.get('total_tokens_per_sec')}`",
                f"- Requests/s: `{summary.get('requests_per_sec')}`",
            ]
        )

    requests = summary.get("requests", [])
    if isinstance(requests, list) and requests:
        lines.extend(["", "## Requests", ""])
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                continue
            lines.extend(
                [
                    f"### Request {index}",
                    "",
                    f"- Sample ID: `{request.get('sample_id', 'unknown')}`",
                    f"- Input tokens: `{request.get('input_tokens', 0)}`",
                    f"- Output tokens: `{request.get('output_tokens', 0)}`",
                    f"- Stop reason: `{request.get('stop_reason', 'unknown')}`",
                    f"- TTFT ms: `{request.get('ttft_ms')}`",
                    f"- TPOT ms: `{request.get('tpot_ms')}`",
                    f"- E2E ms: `{request.get('e2e_ms')}`",
                    "",
                ]
            )

    lines.extend(["", "## Artifacts", ""])
    artifacts = summary.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, path in artifacts.items():
            lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path, summary: dict[str, object]) -> None:
    path.write_text(render_markdown_report(summary), encoding="utf-8")
