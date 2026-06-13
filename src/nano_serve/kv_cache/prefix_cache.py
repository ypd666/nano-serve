"""Block-level prefix cache with a radix lookup index."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrefixCacheEntry:
    key: tuple[int, ...]
    block_ids: tuple[int, ...]
    token_count: int
    last_access_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "block_ids": list(self.block_ids),
            "last_access_ns": self.last_access_ns,
        }


@dataclass(frozen=True)
class PrefixCacheLookup:
    hit: bool
    matched_tokens: int
    block_ids: tuple[int, ...]
    matched_blocks: int
    cpu_lookup_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "hit": self.hit,
            "matched_tokens": self.matched_tokens,
            "block_ids": list(self.block_ids),
            "matched_blocks": self.matched_blocks,
            "cpu_lookup_time_ms": self.cpu_lookup_time_ms,
        }


@dataclass(frozen=True)
class PrefixCacheStats:
    entries: int
    cached_block_refs: int
    cached_entry_tokens: int
    lookup_count: int
    hit_count: int
    hit_tokens: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.lookup_count if self.lookup_count else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": self.entries,
            "cached_block_refs": self.cached_block_refs,
            "cached_entry_tokens": self.cached_entry_tokens,
            "lookup_count": self.lookup_count,
            "hit_count": self.hit_count,
            "hit_tokens": self.hit_tokens,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
        }


@dataclass(frozen=True)
class PrefixCacheInsertResult:
    inserted: tuple[PrefixCacheEntry, ...]
    evicted: tuple[PrefixCacheEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "inserted_entries": len(self.inserted),
            "evicted_entries": len(self.evicted),
            "inserted_tokens": sum(entry.token_count for entry in self.inserted),
            "evicted_tokens": sum(entry.token_count for entry in self.evicted),
        }


@dataclass
class _RadixNode:
    children: dict[int, "_RadixNode"] = field(default_factory=dict)
    entry_key: tuple[int, ...] | None = None


class PrefixCache:
    def __init__(
        self,
        *,
        block_size: int = 16,
        max_entries: int | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.block_size = block_size
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], PrefixCacheEntry] = OrderedDict()
        self._root = _RadixNode()
        self.lookup_count = 0
        self.hit_count = 0
        self.hit_tokens = 0
        self.evictions = 0

    def lookup(self, token_ids: list[int]) -> PrefixCacheLookup:
        start_ns = time.monotonic_ns()
        self.lookup_count += 1
        full_block_tokens = _full_block_prefix(token_ids, self.block_size)
        entry = self._longest_entry(full_block_tokens)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        if entry is None:
            return PrefixCacheLookup(
                hit=False,
                matched_tokens=0,
                block_ids=(),
                matched_blocks=0,
                cpu_lookup_time_ms=elapsed_ms,
            )
        refreshed = PrefixCacheEntry(
            key=entry.key,
            block_ids=entry.block_ids,
            token_count=entry.token_count,
            last_access_ns=time.monotonic_ns(),
        )
        self._entries[entry.key] = refreshed
        self._entries.move_to_end(entry.key)
        self.hit_count += 1
        self.hit_tokens += entry.token_count
        return PrefixCacheLookup(
            hit=True,
            matched_tokens=entry.token_count,
            block_ids=entry.block_ids,
            matched_blocks=len(entry.block_ids),
            cpu_lookup_time_ms=elapsed_ms,
        )

    def insert(
        self,
        token_ids: list[int],
        block_ids: list[int],
        *,
        on_insert: Callable[[PrefixCacheEntry], None] | None = None,
        on_evict: Callable[[PrefixCacheEntry], None] | None = None,
    ) -> PrefixCacheInsertResult:
        full_key = _full_block_prefix(token_ids, self.block_size)
        if not full_key:
            return PrefixCacheInsertResult(inserted=(), evicted=())
        expected_blocks = len(full_key) // self.block_size
        if len(block_ids) < expected_blocks:
            raise ValueError(
                f"expected at least {expected_blocks} blocks, got {len(block_ids)}"
            )

        inserted: list[PrefixCacheEntry] = []
        evicted: list[PrefixCacheEntry] = []
        for block_count in range(1, expected_blocks + 1):
            token_count = block_count * self.block_size
            key = tuple(token_ids[:token_count])
            cached_blocks = tuple(block_ids[:block_count])
            existing = self._entries.pop(key, None)
            if existing is not None:
                if existing.block_ids == cached_blocks:
                    refreshed = PrefixCacheEntry(
                        key=existing.key,
                        block_ids=existing.block_ids,
                        token_count=existing.token_count,
                        last_access_ns=time.monotonic_ns(),
                    )
                    self._entries[key] = refreshed
                    self._entries.move_to_end(key)
                    continue
                self._remove_radix(key)
                evicted.append(existing)
                if on_evict is not None:
                    on_evict(existing)
            entry = PrefixCacheEntry(
                key=key,
                block_ids=cached_blocks,
                token_count=token_count,
                last_access_ns=time.monotonic_ns(),
            )
            self._entries[key] = entry
            self._insert_radix(key)
            inserted.append(entry)
            if on_insert is not None:
                on_insert(entry)

        evicted.extend(self._evict_if_needed(on_evict=on_evict))
        return PrefixCacheInsertResult(
            inserted=tuple(inserted),
            evicted=tuple(evicted),
        )

    def stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(
            entries=len(self._entries),
            cached_block_refs=sum(len(entry.block_ids) for entry in self._entries.values()),
            cached_entry_tokens=sum(entry.token_count for entry in self._entries.values()),
            lookup_count=self.lookup_count,
            hit_count=self.hit_count,
            hit_tokens=self.hit_tokens,
            evictions=self.evictions,
        )

    def entries(self) -> list[PrefixCacheEntry]:
        return list(self._entries.values())

    def _longest_entry(self, token_ids: tuple[int, ...]) -> PrefixCacheEntry | None:
        node = self._root
        best_key = node.entry_key
        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                break
            node = child
            if node.entry_key is not None:
                best_key = node.entry_key
        if best_key is None:
            return None
        return self._entries.get(best_key)

    def _insert_radix(self, key: tuple[int, ...]) -> None:
        node = self._root
        for token_id in key:
            node = node.children.setdefault(token_id, _RadixNode())
        node.entry_key = key

    def _remove_radix(self, key: tuple[int, ...]) -> None:
        path: list[tuple[_RadixNode, int]] = []
        node = self._root
        for token_id in key:
            child = node.children.get(token_id)
            if child is None:
                return
            path.append((node, token_id))
            node = child
        node.entry_key = None
        while path and not node.children and node.entry_key is None:
            parent, token_id = path.pop()
            parent.children.pop(token_id, None)
            node = parent

    def _evict_if_needed(
        self,
        *,
        on_evict: Callable[[PrefixCacheEntry], None] | None,
    ) -> list[PrefixCacheEntry]:
        evicted: list[PrefixCacheEntry] = []
        while self.max_entries is not None and len(self._entries) > self.max_entries:
            _, entry = self._entries.popitem(last=False)
            self._remove_radix(entry.key)
            self.evictions += 1
            evicted.append(entry)
            if on_evict is not None:
                on_evict(entry)
        return evicted


def _full_block_prefix(token_ids: list[int], block_size: int) -> tuple[int, ...]:
    full_tokens = (len(token_ids) // block_size) * block_size
    return tuple(token_ids[:full_tokens])
