"""TileLang availability checks.

TileLang is optional and may be unavailable on local development machines. Keep
all imports guarded so CPU-only paths and macOS development do not fail at
module import time.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class TileLangAvailability:
    available: bool
    version: str | None = None
    error: str | None = None
    probe_mode: str = "subprocess"

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "version": self.version,
            "error": self.error,
            "probe_mode": self.probe_mode,
        }


@lru_cache(maxsize=1)
def check_tilelang_available() -> TileLangAvailability:
    if os.environ.get("NANO_SERVE_TILELANG_AVAILABILITY_MODE") == "in_process":
        return _check_tilelang_available_in_process()
    return _check_tilelang_available_subprocess()


def _check_tilelang_available_in_process() -> TileLangAvailability:
    try:
        tilelang = importlib.import_module("tilelang")
    except Exception as exc:  # pragma: no cover - depends on local optional deps.
        return TileLangAvailability(
            available=False,
            error=f"{type(exc).__name__}: {exc}",
            probe_mode="in_process",
        )

    version = getattr(tilelang, "__version__", None)
    return TileLangAvailability(
        available=True,
        version=str(version) if version is not None else None,
        probe_mode="in_process",
    )


def _check_tilelang_available_subprocess() -> TileLangAvailability:
    code = (
        "import json\n"
        "try:\n"
        "    import tilelang\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'available': False, 'version': None, "
        "'error': f'{type(exc).__name__}: {exc}'}))\n"
        "else:\n"
        "    print(json.dumps({'available': True, "
        "'version': str(getattr(tilelang, '__version__', 'unknown')), "
        "'error': None}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        return TileLangAvailability(
            available=False,
            error=message or f"tilelang probe exited with {result.returncode}",
            probe_mode="subprocess",
        )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return TileLangAvailability(
            available=False,
            error=f"invalid tilelang probe output: {exc}",
            probe_mode="subprocess",
        )
    return TileLangAvailability(
        available=bool(payload.get("available")),
        version=str(payload["version"]) if payload.get("version") is not None else None,
        error=str(payload["error"]) if payload.get("error") is not None else None,
        probe_mode="subprocess",
    )
