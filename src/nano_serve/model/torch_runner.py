"""Torch model runner placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nano_serve.engine.batch import BatchPlan
from nano_serve.model.loader import ModelLoader, ModelSpec
from nano_serve.model.runner import ModelOutput


@dataclass
class PrefillOutput:
    logits: Any
    metadata: dict[str, Any]


@dataclass
class DecodeOutput:
    logits: Any
    metadata: dict[str, Any]


@dataclass
class TorchModelRunner:
    model: Any

    @classmethod
    def from_model_spec(cls, spec: ModelSpec) -> "TorchModelRunner":
        return cls(model=ModelLoader().load(spec))

    def execute(self, batch: BatchPlan) -> ModelOutput:
        if batch.batch_size != 1:
            raise NotImplementedError("Phase 1 TorchModelRunner only supports batch_size=1.")
        if not batch.input_token_ids or not batch.input_token_ids[0]:
            raise ValueError("input_token_ids must contain one non-empty request.")

        import torch

        token_ids = batch.input_token_ids[0]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        with torch.inference_mode():
            logits = self.model(input_ids)

        return ModelOutput(
            logits=logits,
            metadata={
                "batch_size": batch.batch_size,
                "seq_len": len(token_ids),
                "runner": "torch_full_context",
            },
        )

    def next_token_logits(self, token_ids: list[int]):
        if not token_ids:
            raise ValueError("token_ids must not be empty")

        import torch

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        with torch.inference_mode():
            return self.model.next_token_logits(input_ids)

    def prefill(self, prompt_token_ids: list[int]) -> PrefillOutput:
        logits = self.next_token_logits(prompt_token_ids)
        return PrefillOutput(
            logits=logits,
            metadata={
                "phase": "prefill",
                "input_tokens": len(prompt_token_ids),
                "runner": "torch_full_context",
                "kv_cache": "none",
            },
        )

    def decode(self, context_token_ids: list[int], *, new_token_id: int | None = None) -> DecodeOutput:
        logits = self.next_token_logits(context_token_ids)
        return DecodeOutput(
            logits=logits,
            metadata={
                "phase": "decode",
                "context_tokens": len(context_token_ids),
                "new_token_id": new_token_id,
                "runner": "torch_full_context",
                "kv_cache": "none",
            },
        )

