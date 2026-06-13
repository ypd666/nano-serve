"""CPU-side overlap helpers used by Phase 10 experiments."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, order=True)
class ShapeBucket:
    batch_size: int
    seq_len: int

    @property
    def capacity(self) -> int:
        return self.batch_size * self.seq_len

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "capacity": self.capacity,
        }


@dataclass(frozen=True)
class ShapeBucketSelection:
    requested_batch_size: int
    requested_seq_len: int
    bucket: ShapeBucket
    padded_elements: int

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_batch_size": self.requested_batch_size,
            "requested_seq_len": self.requested_seq_len,
            "bucket": self.bucket.to_dict(),
            "padded_elements": self.padded_elements,
        }


class ShapeBucketSelector:
    def __init__(self, buckets: list[ShapeBucket]) -> None:
        if not buckets:
            raise ValueError("buckets must not be empty")
        if any(bucket.batch_size <= 0 or bucket.seq_len <= 0 for bucket in buckets):
            raise ValueError("bucket dimensions must be positive")
        self.buckets = sorted(set(buckets), key=lambda bucket: (bucket.capacity, bucket))

    def select(self, *, batch_size: int, seq_len: int) -> ShapeBucketSelection:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("requested shape dimensions must be positive")
        for bucket in self.buckets:
            if bucket.batch_size >= batch_size and bucket.seq_len >= seq_len:
                return ShapeBucketSelection(
                    requested_batch_size=batch_size,
                    requested_seq_len=seq_len,
                    bucket=bucket,
                    padded_elements=bucket.capacity - batch_size * seq_len,
                )
        raise ValueError(f"no shape bucket can fit batch={batch_size}, seq_len={seq_len}")


@dataclass(frozen=True)
class BatchMetadata:
    iteration: int
    request_ids: tuple[str, ...]
    batch_size: int
    seq_len: int

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "request_ids": list(self.request_ids),
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
        }


@dataclass(frozen=True)
class PublishedBatchMetadata:
    slot: int
    metadata: BatchMetadata
    publish_time_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "publish_time_ns": self.publish_time_ns,
            **self.metadata.to_dict(),
        }


class DoubleBufferedBatchMetadata:
    def __init__(self) -> None:
        self._slots: list[BatchMetadata | None] = [None, None]
        self._next_slot = 0

    def publish(self, metadata: BatchMetadata) -> PublishedBatchMetadata:
        slot = self._next_slot
        self._slots[slot] = metadata
        self._next_slot = 1 - self._next_slot
        return PublishedBatchMetadata(
            slot=slot,
            metadata=metadata,
            publish_time_ns=time.monotonic_ns(),
        )

    def latest(self) -> PublishedBatchMetadata | None:
        previous_slot = 1 - self._next_slot
        metadata = self._slots[previous_slot]
        if metadata is None:
            return None
        return PublishedBatchMetadata(
            slot=previous_slot,
            metadata=metadata,
            publish_time_ns=time.monotonic_ns(),
        )


@dataclass(frozen=True)
class SchedulerPrepResult:
    iteration: int
    request_ids: tuple[str, ...]
    cpu_schedule_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "request_ids": list(self.request_ids),
            "cpu_schedule_time_ms": self.cpu_schedule_time_ms,
        }


class AsyncSchedulerPrep:
    def __init__(self, *, max_workers: int = 1) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        *,
        iteration: int,
        request_ids: list[str],
    ) -> Future[SchedulerPrepResult]:
        def prepare() -> SchedulerPrepResult:
            start_ns = time.monotonic_ns()
            ordered = tuple(sorted(request_ids))
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
            return SchedulerPrepResult(
                iteration=iteration,
                request_ids=ordered,
                cpu_schedule_time_ms=elapsed_ms,
            )

        return self._executor.submit(prepare)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ...


@dataclass(frozen=True)
class TokenizerWorkerResult:
    index: int
    token_ids: list[int]
    tokenize_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "token_ids": list(self.token_ids),
            "num_tokens": len(self.token_ids),
            "tokenize_time_ms": self.tokenize_time_ms,
        }


class TokenizerWorker:
    def __init__(self, tokenizer: TokenizerLike, *, max_workers: int = 1) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.tokenizer = tokenizer
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, *, index: int, text: str) -> Future[TokenizerWorkerResult]:
        def tokenize() -> TokenizerWorkerResult:
            start_ns = time.monotonic_ns()
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
            return TokenizerWorkerResult(
                index=index,
                token_ids=token_ids,
                tokenize_time_ms=elapsed_ms,
            )

        return self._executor.submit(tokenize)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
