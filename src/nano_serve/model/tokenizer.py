"""Text-only tokenizer wrapper for Qwen3.5-4B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast


class _TokenizerLike(Protocol):
    eos_token_id: Any
    pad_token_id: Any

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[Any]:
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        ...


@dataclass
class TokenizerWrapper:
    tokenizer: _TokenizerLike
    model_path: Path

    @classmethod
    def from_pretrained(cls, model_path: Path | str) -> "TokenizerWrapper":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Tokenizer loading requires transformers. Install with `pip install -e .[torch]`."
            ) from exc

        path = Path(model_path).resolve(strict=False)
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=True,
        )
        return cls(tokenizer=cast(_TokenizerLike, tokenizer), model_path=path)

    @property
    def eos_token_id(self) -> int | None:
        return _optional_int(getattr(self.tokenizer, "eos_token_id", None))

    @property
    def pad_token_id(self) -> int | None:
        return _optional_int(getattr(self.tokenizer, "pad_token_id", None))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return [int(token_id) for token_id in token_ids]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=False))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
