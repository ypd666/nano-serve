"""Greedy sampler."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nano_serve.sampling.base import SamplingParams


class GreedySampler:
    def sample(self, logits: Sequence[float] | Any, params: SamplingParams | None = None) -> int:
        del params
        if hasattr(logits, "argmax"):
            return int(logits.argmax(dim=-1).item())
        if not logits:
            raise ValueError("logits must not be empty")
        return max(range(len(logits)), key=lambda idx: logits[idx])

