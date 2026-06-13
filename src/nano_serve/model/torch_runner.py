"""Torch model runner placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nano_serve.engine.batch import BatchPlan
from nano_serve.kv_cache.contiguous import ContiguousKVCache, ContiguousKVCacheConfig
from nano_serve.model.loader import ModelLoader, ModelSpec
from nano_serve.model.runner import ModelOutput


@dataclass
class PrefillOutput:
    logits: Any
    metadata: dict[str, Any]
    request_id: str | None = None


@dataclass
class DecodeOutput:
    logits: Any
    metadata: dict[str, Any]
    request_id: str | None = None


@dataclass
class TorchModelRunner:
    model: Any
    kv_cache: ContiguousKVCache | None = None

    @classmethod
    def from_model_spec(
        cls,
        spec: ModelSpec,
        *,
        kv_cache: str = "none",
        block_size: int = 16,
    ) -> "TorchModelRunner":
        model = ModelLoader().load(spec)
        cache = None
        if kv_cache == "contiguous":
            cache = ContiguousKVCache(
                ContiguousKVCacheConfig(
                    max_model_len=spec.max_model_len or model.config.max_position_embeddings,
                    num_layers=model.config.num_hidden_layers,
                    block_size=block_size,
                )
            )
        elif kv_cache != "none":
            raise ValueError(f"Unsupported torch runner KV cache: {kv_cache}")
        return cls(model=model, kv_cache=cache)

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

    def prefill(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str | None = None,
        max_decode_tokens: int = 0,
    ) -> PrefillOutput:
        if self.kv_cache is None:
            logits = self.next_token_logits(prompt_token_ids)
            kv_metadata: dict[str, Any] = {
                "kv_cache": "none",
                "kv_sequence_length": 0,
                "kv_bytes_used": 0,
            }
        else:
            if request_id is None:
                raise ValueError("request_id is required when kv_cache=contiguous")
            import torch

            input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=self.model.device)
            with torch.inference_mode():
                output = self.model.prefill_with_cache(input_ids)
            self.kv_cache.allocate_prefill(
                request_id,
                len(prompt_token_ids),
                max_decode_tokens=max_decode_tokens,
                layer_states=output.layer_states,
            )
            logits = output.logits
            stats = self.kv_cache.stats()
            kv_metadata = {
                "kv_cache": "contiguous",
                "kv_sequence_length": self.kv_cache.sequence_length(request_id),
                "kv_bytes_used": stats.bytes_used,
                "kv_blocks_used": len(self.kv_cache.get_block_table(request_id)),
                "kv_fragmentation": stats.fragmentation,
            }
        return PrefillOutput(
            logits=logits,
            metadata={
                "phase": "prefill",
                "input_tokens": len(prompt_token_ids),
                "runner": "torch_cached" if self.kv_cache is not None else "torch_full_context",
                **kv_metadata,
            },
            request_id=request_id,
        )

    def decode(
        self,
        context_token_ids: list[int],
        *,
        new_token_id: int | None = None,
        request_id: str | None = None,
    ) -> DecodeOutput:
        if self.kv_cache is None:
            logits = self.next_token_logits(context_token_ids)
            kv_metadata: dict[str, Any] = {
                "kv_cache": "none",
                "kv_sequence_length": 0,
                "kv_bytes_used": 0,
            }
        else:
            if request_id is None:
                raise ValueError("request_id is required when kv_cache=contiguous")
            if new_token_id is None:
                raise ValueError("new_token_id is required for cached decode")
            import torch

            position_offset = self.kv_cache.sequence_length(request_id)
            input_ids = torch.tensor([[new_token_id]], dtype=torch.long, device=self.model.device)
            with torch.inference_mode():
                output = self.model.decode_with_cache(
                    input_ids,
                    layer_states=self.kv_cache.layer_states(request_id),
                    position_offset=position_offset,
                )
            self.kv_cache.allocate_decode_slot(request_id)
            self.kv_cache.set_layer_states(
                request_id,
                output.layer_states,
                sequence_length=position_offset + 1,
            )
            logits = output.logits
            stats = self.kv_cache.stats()
            kv_metadata = {
                "kv_cache": "contiguous",
                "kv_sequence_length": self.kv_cache.sequence_length(request_id),
                "kv_bytes_used": stats.bytes_used,
                "kv_blocks_used": len(self.kv_cache.get_block_table(request_id)),
                "kv_fragmentation": stats.fragmentation,
            }
        return DecodeOutput(
            logits=logits,
            metadata={
                "phase": "decode",
                "context_tokens": len(context_token_ids),
                "new_token_id": new_token_id,
                "runner": "torch_cached" if self.kv_cache is not None else "torch_full_context",
                **kv_metadata,
            },
            request_id=request_id,
        )

    def free(self, request_id: str) -> None:
        if self.kv_cache is not None:
            self.kv_cache.free(request_id)

