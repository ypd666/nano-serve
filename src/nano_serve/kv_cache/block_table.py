"""Paged KV block table data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KVBlock:
    block_id: int
    ref_count: int = 0
    used_tokens: int = 0


@dataclass
class BlockTable:
    request_id: str
    block_ids: list[int] = field(default_factory=list)
    seq_len: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "block_ids": list(self.block_ids),
            "seq_len": self.seq_len,
        }
