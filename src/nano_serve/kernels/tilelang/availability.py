"""TileLang availability checks.

TileLang is optional and may be unavailable on local development machines. Keep
all imports guarded so CPU-only paths and macOS development do not fail at
module import time.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class TileLangAvailability:
    available: bool
    version: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "version": self.version,
            "error": self.error,
        }


@lru_cache(maxsize=1)
def check_tilelang_available() -> TileLangAvailability:
    try:
        tilelang = importlib.import_module("tilelang")
    except Exception as exc:  # pragma: no cover - depends on local optional deps.
        return TileLangAvailability(available=False, error=f"{type(exc).__name__}: {exc}")

    version = getattr(tilelang, "__version__", None)
    return TileLangAvailability(
        available=True,
        version=str(version) if version is not None else None,
    )
