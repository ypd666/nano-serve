"""Draft model interfaces and deterministic reference drafts."""

from __future__ import annotations

from typing import Protocol


class DraftModel(Protocol):
    def propose(self, context_token_ids: list[int], *, max_tokens: int) -> list[int]:
        ...


class StaticDraftModel:
    def __init__(self, token_stream: list[int], *, base_context_len: int = 0) -> None:
        self.token_stream = list(token_stream)
        self.base_context_len = base_context_len

    def propose(self, context_token_ids: list[int], *, max_tokens: int) -> list[int]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        start = max(0, len(context_token_ids) - self.base_context_len)
        return self.token_stream[start : start + max_tokens]
