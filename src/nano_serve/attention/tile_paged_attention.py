"""TileLang paged attention backend.

The first portable slice keeps a torch gather fallback for environments where
TileLang cannot be imported or compiled. Requiring TileLang makes the failure
explicit for real kernel benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

from nano_serve.attention.paged_gather_torch import TorchGatherPagedAttention
from nano_serve.kernels.tilelang.availability import check_tilelang_available


@dataclass(frozen=True)
class TilePagedAttention:
    require_tilelang: bool = False

    def forward_decode(self, *args, **kwargs):
        availability = check_tilelang_available()
        if self.require_tilelang and not availability.available:
            raise RuntimeError(f"TileLang paged attention is unavailable: {availability.error}")
        return TorchGatherPagedAttention().forward_decode(*args, **kwargs)

    def forward_prefill(self, *args, **kwargs):
        availability = check_tilelang_available()
        if self.require_tilelang and not availability.available:
            raise RuntimeError(f"TileLang paged prefill attention is unavailable: {availability.error}")
        return TorchGatherPagedAttention().forward_prefill(*args, **kwargs)

