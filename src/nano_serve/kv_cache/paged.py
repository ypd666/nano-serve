"""Paged KV allocator for fixed-size cache blocks."""

from __future__ import annotations

from dataclasses import dataclass

from nano_serve.kv_cache.base import KVHandle
from nano_serve.kv_cache.block_table import BlockTable, KVBlock


@dataclass(frozen=True)
class PagedKVStats:
    num_blocks: int
    used_blocks: int
    free_blocks: int
    used_tokens: int
    token_capacity: int
    internal_fragmentation: float
    oom_count: int
    resident_requests: int
    max_resident_requests: int

    def to_dict(self) -> dict[str, object]:
        return {
            "num_blocks": self.num_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "used_tokens": self.used_tokens,
            "token_capacity": self.token_capacity,
            "internal_fragmentation": self.internal_fragmentation,
            "oom_count": self.oom_count,
            "resident_requests": self.resident_requests,
            "max_resident_requests": self.max_resident_requests,
        }


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [KVBlock(block_id=i) for i in range(num_blocks)]
        self.free_blocks: list[int] = list(range(num_blocks))
        self.tables: dict[str, BlockTable] = {}
        self.oom_count = 0
        self.max_resident_requests = 0

    def allocate_prefill(
        self,
        request_id: str,
        n_tokens: int,
        *,
        max_decode_tokens: int = 0,
    ) -> KVHandle:
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        if max_decode_tokens < 0:
            raise ValueError("max_decode_tokens must be non-negative")
        existing = self.tables.get(request_id)
        if existing is not None:
            self.free(request_id)
        needed = _ceil_div(n_tokens, self.block_size)
        self._reserve_blocks(needed)
        table = BlockTable(request_id=request_id)
        self.tables[request_id] = table
        while len(table.block_ids) < _ceil_div(n_tokens, self.block_size):
            table.block_ids.append(self._alloc_block())
        table.seq_len = n_tokens
        self._update_block_usage(table)
        self.max_resident_requests = max(self.max_resident_requests, len(self.tables))
        return KVHandle(
            request_id=request_id,
            num_tokens=n_tokens,
            block_ids=list(table.block_ids),
        )

    def allocate_decode_slot(self, request_id: str) -> KVHandle:
        table = self._table(request_id)
        if table.seq_len % self.block_size == 0:
            table.block_ids.append(self._alloc_block())
        table.seq_len += 1
        self._update_block_usage(table)
        return KVHandle(
            request_id=request_id,
            num_tokens=table.seq_len,
            block_ids=list(table.block_ids),
        )

    def free(self, request_id: str) -> None:
        table = self.tables.pop(request_id, None)
        if table is None:
            return
        for block_id in table.block_ids:
            self.blocks[block_id].ref_count = 0
            self.blocks[block_id].used_tokens = 0
            self.free_blocks.append(block_id)
        self.free_blocks.sort()

    def get_block_table(self, request_id: str) -> list[int]:
        table = self.tables.get(request_id)
        return [] if table is None else list(table.block_ids)

    def stats(self) -> PagedKVStats:
        used_tokens = sum(table.seq_len for table in self.tables.values())
        used_blocks = self.num_blocks - len(self.free_blocks)
        token_capacity = used_blocks * self.block_size
        fragmentation = 0.0
        if token_capacity:
            fragmentation = 1.0 - (used_tokens / token_capacity)
        return PagedKVStats(
            num_blocks=self.num_blocks,
            used_blocks=used_blocks,
            free_blocks=len(self.free_blocks),
            used_tokens=used_tokens,
            token_capacity=token_capacity,
            internal_fragmentation=fragmentation,
            oom_count=self.oom_count,
            resident_requests=len(self.tables),
            max_resident_requests=self.max_resident_requests,
        )

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            self.oom_count += 1
            raise MemoryError("paged KV cache is out of blocks")
        block_id = self.free_blocks.pop(0)
        self.blocks[block_id].ref_count = 1
        return block_id

    def _reserve_blocks(self, needed: int) -> None:
        if len(self.free_blocks) < needed:
            self.oom_count += 1
            raise MemoryError(
                f"paged KV cache is out of blocks: need {needed}, free {len(self.free_blocks)}"
            )

    def _table(self, request_id: str) -> BlockTable:
        try:
            return self.tables[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown paged KV request: {request_id}") from exc

    def _update_block_usage(self, table: BlockTable) -> None:
        remaining = table.seq_len
        for block_id in table.block_ids:
            used = min(self.block_size, max(remaining, 0))
            self.blocks[block_id].used_tokens = used
            self.blocks[block_id].ref_count = 1
            remaining -= used


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
