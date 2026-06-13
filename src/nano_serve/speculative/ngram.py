"""N-gram speculation reference implementation."""

from __future__ import annotations


class NGramSpeculator:
    def __init__(self, *, ngram_size: int = 2) -> None:
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        self.ngram_size = ngram_size

    def propose(self, context_token_ids: list[int], *, max_tokens: int) -> list[int]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if len(context_token_ids) < self.ngram_size:
            return []
        suffix = context_token_ids[-self.ngram_size :]
        for start in range(len(context_token_ids) - self.ngram_size):
            if context_token_ids[start : start + self.ngram_size] == suffix:
                continuation_start = start + self.ngram_size
                continuation_end = min(continuation_start + max_tokens, len(context_token_ids))
                return context_token_ids[continuation_start:continuation_end]
        return []
