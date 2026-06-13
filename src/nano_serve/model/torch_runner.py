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

    def next_token_logits_batch(
        self,
        token_ids_batch: list[list[int]],
        *,
        pad_token_id: int = 0,
    ):
        if not token_ids_batch:
            raise ValueError("token_ids_batch must not be empty")
        if any(not token_ids for token_ids in token_ids_batch):
            raise ValueError("token_ids_batch must contain non-empty requests")

        import torch

        lengths = torch.tensor(
            [len(token_ids) for token_ids in token_ids_batch],
            dtype=torch.long,
            device=self.model.device,
        )
        max_len = int(lengths.max().item())
        input_ids = torch.full(
            (len(token_ids_batch), max_len),
            int(pad_token_id),
            dtype=torch.long,
            device=self.model.device,
        )
        for index, token_ids in enumerate(token_ids_batch):
            input_ids[index, : len(token_ids)] = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=self.model.device,
            )

        with torch.inference_mode():
            logits = self.model(input_ids)
        batch_indices = torch.arange(len(token_ids_batch), device=self.model.device)
        return logits[batch_indices, lengths - 1, :]

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

    def prefill_chunk(
        self,
        prompt_token_ids: list[int],
        *,
        start: int,
        end: int,
        request_id: str | None = None,
        max_decode_tokens: int = 0,
        final_chunk: bool = False,
    ) -> PrefillOutput:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if start < 0 or end <= start or end > len(prompt_token_ids):
            raise ValueError(f"invalid prefill chunk range: {start}:{end}")

        chunk_token_ids = prompt_token_ids[start:end]
        if self.kv_cache is None:
            logits = self.next_token_logits(prompt_token_ids[:end]) if final_chunk else None
            kv_metadata: dict[str, Any] = {
                "kv_cache": "none",
                "kv_sequence_length": 0,
                "kv_bytes_used": 0,
            }
        else:
            if request_id is None:
                raise ValueError("request_id is required when kv_cache=contiguous")
            import torch

            if start == 0:
                input_ids = torch.tensor(
                    [chunk_token_ids],
                    dtype=torch.long,
                    device=self.model.device,
                )
                with torch.inference_mode():
                    output = self.model.prefill_with_cache(input_ids)
                remaining_prompt_tokens = len(prompt_token_ids) - end
                self.kv_cache.allocate_prefill(
                    request_id,
                    len(chunk_token_ids),
                    max_decode_tokens=remaining_prompt_tokens + max_decode_tokens,
                    layer_states=output.layer_states,
                )
                logits = output.logits if final_chunk else None
            else:
                if self.kv_cache.sequence_length(request_id) != start:
                    raise ValueError(
                        "prefill chunk start must match cached sequence length: "
                        f"{start} != {self.kv_cache.sequence_length(request_id)}"
                    )
                output = None
                for token_id in chunk_token_ids:
                    position_offset = self.kv_cache.sequence_length(request_id)
                    input_ids = torch.tensor(
                        [[token_id]],
                        dtype=torch.long,
                        device=self.model.device,
                    )
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
                if output is None:
                    raise RuntimeError("prefill chunk produced no output")
                logits = output.logits if final_chunk else None

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
                "phase": "prefill_chunk",
                "chunk_start": start,
                "chunk_end": end,
                "chunk_tokens": end - start,
                "input_tokens": end,
                "total_prompt_tokens": len(prompt_token_ids),
                "final_chunk": final_chunk,
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

