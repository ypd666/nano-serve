"""Runtime event schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import time_ns
from typing import Any

from nano_serve.platform import PlatformInfo, detect_platform


@dataclass(frozen=True)
class Event:
    name: str
    timestamp_ns: int = field(default_factory=time_ns)
    fields: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "timestamp_ns": self.timestamp_ns,
            "fields": _jsonable(self.fields),
        }


def platform_event(info: PlatformInfo | None = None) -> Event:
    info = info or detect_platform()
    return Event(name="platform_detected", fields=info.to_log_fields())


class JSONLEventWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, event: Event) -> None:
        self._file.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JSONLEventWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_jsonl_events(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
