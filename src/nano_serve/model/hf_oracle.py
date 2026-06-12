"""Hugging Face correctness oracle.

This module is intentionally an oracle, not the serving engine. It exists so
Phase 1 and later implementations can compare logits against a trusted local
reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast


class _TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[Any]:
        ...


class _ModelOutputLike(Protocol):
    logits: Any


class _ModelLike(Protocol):
    def to(self, device: Any) -> "_ModelLike":
        ...

    def eval(self) -> "_ModelLike":
        ...

    def __call__(self, *, input_ids: Any, use_cache: bool = False) -> _ModelOutputLike:
        ...


@dataclass
class HuggingFaceOracle:
    model: _ModelLike
    tokenizer: _TokenizerLike
    device: Any

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path | str,
        *,
        device: str = "cuda",
        dtype: str = "auto",
    ) -> "HuggingFaceOracle":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFaceOracle requires torch and transformers. "
                "Install with `pip install -e .[torch]`."
            ) from exc

        path = Path(model_path).resolve(strict=False)
        torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        torch_dtype = _parse_dtype(dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=True,
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(path),
            dtype=torch_dtype,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = cast(_ModelLike, hf_model).to(torch_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=cast(_TokenizerLike, tokenizer),
            device=torch_device,
        )

    def encode(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.tokenizer.encode(text, add_special_tokens=False)]

    def logits(self, token_ids: list[int]):
        import torch

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, use_cache=False)
        return output.logits

    def next_token_logits(self, token_ids: list[int]):
        return self.logits(token_ids)[:, -1, :]


def _parse_dtype(dtype: str):
    if dtype == "auto":
        return "auto"

    import torch

    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")
