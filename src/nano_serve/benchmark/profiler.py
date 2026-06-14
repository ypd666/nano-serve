"""Profiler hooks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from nano_serve.observability.tracing import nvtx_range as _nvtx_range


@contextmanager
def nvtx_range(name: str, *, enabled: bool = True) -> Iterator[None]:
    with _nvtx_range(name, enabled=enabled):
        yield


def nvtx_label(phase: str, stage: str, **fields: object) -> str:
    label = f"nano_serve.{phase}.{stage}"
    clean_fields = {
        key: value
        for key, value in fields.items()
        if value is not None
    }
    if not clean_fields:
        return label
    metadata = ",".join(
        f"{key}={_format_nvtx_value(value)}"
        for key, value in sorted(clean_fields.items())
    )
    return f"{label}:{metadata}"


def _format_nvtx_value(value: object) -> str:
    text = str(value)
    return text.replace(" ", "_").replace(",", "_").replace(":", "_")
