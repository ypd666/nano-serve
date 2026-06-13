"""Reference LoRA adapter utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoRAAdapter:
    adapter_id: str
    a: Any
    b: Any
    alpha: float = 1.0

    @property
    def rank(self) -> int:
        return int(self.a.shape[-1])

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def apply(self, x: Any) -> Any:
        return (x @ self.a @ self.b) * self.scale


class LoRAAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, LoRAAdapter] = {}

    def register(self, adapter: LoRAAdapter) -> None:
        if adapter.adapter_id in self._adapters:
            raise ValueError(f"duplicate LoRA adapter: {adapter.adapter_id}")
        self._adapters[adapter.adapter_id] = adapter

    def get(self, adapter_id: str) -> LoRAAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise KeyError(f"unknown LoRA adapter: {adapter_id}") from exc

    def apply_batch(self, x: Any, adapter_ids: list[str]) -> Any:
        import torch

        if len(adapter_ids) != x.shape[0]:
            raise ValueError("adapter_ids length must match batch size")
        outputs = []
        for row, adapter_id in enumerate(adapter_ids):
            adapter = self.get(adapter_id)
            outputs.append(x[row : row + 1] + adapter.apply(x[row : row + 1]))
        return torch.cat(outputs, dim=0)

    def switch_count(self, adapter_ids: list[str]) -> int:
        if not adapter_ids:
            return 0
        switches = 0
        previous = adapter_ids[0]
        for adapter_id in adapter_ids[1:]:
            if adapter_id != previous:
                switches += 1
            previous = adapter_id
        return switches

    def __len__(self) -> int:
        return len(self._adapters)
