"""Speculative verification data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VerificationResult:
    accepted_token_ids: list[int]
    emitted_token_ids: list[int]
    rejected_token_id: int | None = None
    replacement_token_id: int | None = None
    bonus_token_id: int | None = None
    target_calls: int = 1
    draft_tokens_proposed: int = 0
    accepted_tokens: int = 0
    rollback_tokens: int = 0
    kv_tokens_appended: int = 0

    @property
    def rejected(self) -> bool:
        return self.rejected_token_id is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted_token_ids": list(self.accepted_token_ids),
            "emitted_token_ids": list(self.emitted_token_ids),
            "rejected_token_id": self.rejected_token_id,
            "replacement_token_id": self.replacement_token_id,
            "bonus_token_id": self.bonus_token_id,
            "target_calls": self.target_calls,
            "draft_tokens_proposed": self.draft_tokens_proposed,
            "accepted_tokens": self.accepted_tokens,
            "rollback_tokens": self.rollback_tokens,
            "kv_tokens_appended": self.kv_tokens_appended,
        }


class Verifier(Protocol):
    def verify(
        self,
        context_token_ids: list[int],
        draft_token_ids: list[int],
    ) -> VerificationResult:
        ...


class GreedyTokenStreamVerifier:
    def __init__(self, target_stream: list[int], *, base_context_len: int = 0) -> None:
        self.target_stream = list(target_stream)
        self.base_context_len = base_context_len

    def verify(
        self,
        context_token_ids: list[int],
        draft_token_ids: list[int],
    ) -> VerificationResult:
        generated_offset = max(0, len(context_token_ids) - self.base_context_len)
        accepted: list[int] = []
        emitted: list[int] = []
        for index, draft_token_id in enumerate(draft_token_ids):
            target_index = generated_offset + index
            if target_index >= len(self.target_stream):
                break
            target_token_id = self.target_stream[target_index]
            if draft_token_id != target_token_id:
                emitted = [*accepted, target_token_id]
                return VerificationResult(
                    accepted_token_ids=accepted,
                    emitted_token_ids=emitted,
                    rejected_token_id=draft_token_id,
                    replacement_token_id=target_token_id,
                    target_calls=1,
                    draft_tokens_proposed=len(draft_token_ids),
                    accepted_tokens=len(accepted),
                    rollback_tokens=len(draft_token_ids) - len(accepted),
                    kv_tokens_appended=len(emitted),
                )
            accepted.append(draft_token_id)

        bonus_index = generated_offset + len(accepted)
        bonus_token_id = (
            self.target_stream[bonus_index] if bonus_index < len(self.target_stream) else None
        )
        emitted = list(accepted)
        if bonus_token_id is not None:
            emitted.append(bonus_token_id)
        return VerificationResult(
            accepted_token_ids=accepted,
            emitted_token_ids=emitted,
            bonus_token_id=bonus_token_id,
            target_calls=1,
            draft_tokens_proposed=len(draft_token_ids),
            accepted_tokens=len(accepted),
            rollback_tokens=0,
            kv_tokens_appended=len(emitted),
        )
