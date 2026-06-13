from __future__ import annotations

import pytest


def test_contiguous_kv_prefill_roundtrip() -> None:
    import torch

    cache = _make_contiguous_cache(max_tokens=8)
    handle = cache.allocate_prefill("req-a", 3, max_decode_tokens=2)
    keys, values = _make_kv(num_tokens=3)

    cache.write_prefill("req-a", keys, values)
    actual_keys, actual_values = cache.get_kv("req-a")

    assert handle.request_id == "req-a"
    assert handle.num_tokens == 3
    assert actual_keys.shape == (2, 3, 2, 3)
    assert actual_values.shape == (2, 3, 2, 3)
    torch.testing.assert_close(actual_keys, keys)
    torch.testing.assert_close(actual_values, values)


def test_contiguous_kv_decode_append() -> None:
    import torch

    cache = _make_contiguous_cache(max_tokens=8)
    prefill_keys, prefill_values = _make_kv(num_tokens=2)
    decode_keys, decode_values = _make_kv(num_tokens=1, offset=100)

    cache.allocate_prefill("req-a", 2, max_decode_tokens=2)
    cache.write_prefill("req-a", prefill_keys, prefill_values)
    handle = cache.append_decode("req-a", decode_keys, decode_values)
    actual_keys, actual_values = cache.get_kv("req-a")

    assert handle.num_tokens == 3
    assert actual_keys.shape == (2, 3, 2, 3)
    torch.testing.assert_close(actual_keys[:, :2], prefill_keys)
    torch.testing.assert_close(actual_values[:, :2], prefill_values)
    torch.testing.assert_close(actual_keys[:, 2:3], decode_keys)
    torch.testing.assert_close(actual_values[:, 2:3], decode_values)


def test_contiguous_kv_decode_append_capacity_overflow() -> None:
    cache = _make_contiguous_cache(max_tokens=4)
    prefill_keys, prefill_values = _make_kv(num_tokens=2)
    decode_keys, decode_values = _make_kv(num_tokens=1, offset=100)

    cache.allocate_prefill("req-a", 2, max_decode_tokens=0)
    cache.write_prefill("req-a", prefill_keys, prefill_values)

    with pytest.raises(ValueError, match="capacity"):
        cache.append_decode("req-a", decode_keys, decode_values)


def test_contiguous_kv_global_capacity_overflow() -> None:
    cache = _make_contiguous_cache(max_tokens=3)

    cache.allocate_prefill("req-a", 2, max_decode_tokens=1)

    with pytest.raises(MemoryError, match="out of contiguous KV capacity"):
        cache.allocate_prefill("req-b", 1, max_decode_tokens=0)


def test_contiguous_kv_free_removes_request() -> None:
    cache = _make_contiguous_cache(max_tokens=4)
    keys, values = _make_kv(num_tokens=2)

    cache.allocate_prefill("req-a", 2, max_decode_tokens=1)
    cache.write_prefill("req-a", keys, values)
    cache.free("req-a")

    assert cache.get_block_table("req-a") == []
    with pytest.raises(KeyError, match="req-a"):
        cache.get_kv("req-a")


def test_contiguous_kv_rejects_wrong_prefill_shape() -> None:
    import torch

    cache = _make_contiguous_cache(max_tokens=4)
    keys, values = _make_kv(num_tokens=2)
    bad_keys = torch.zeros((2, 2, 2, 4), dtype=torch.float32)

    cache.allocate_prefill("req-a", 2, max_decode_tokens=1)

    with pytest.raises(ValueError, match="shape"):
        cache.write_prefill("req-a", bad_keys, values)


def test_paged_kv_prefill_append_free_and_reuse() -> None:
    from nano_serve.kv_cache.paged import PagedKVCache

    cache = PagedKVCache(num_blocks=4, block_size=2)
    handle = cache.allocate_prefill("req-a", 3)

    assert handle.block_ids == [0, 1]
    assert cache.get_block_table("req-a") == [0, 1]
    assert cache.blocks[0].used_tokens == 2
    assert cache.blocks[1].used_tokens == 1

    handle = cache.allocate_decode_slot("req-a")

    assert handle.num_tokens == 4
    assert handle.block_ids == [0, 1]
    assert cache.blocks[1].used_tokens == 2

    handle = cache.allocate_decode_slot("req-a")

    assert handle.num_tokens == 5
    assert handle.block_ids == [0, 1, 2]
    assert cache.blocks[2].used_tokens == 1
    assert cache.stats().internal_fragmentation == pytest.approx(1 / 6)

    cache.free("req-a")

    assert cache.get_block_table("req-a") == []
    assert cache.stats().free_blocks == 4
    assert cache.allocate_prefill("req-b", 1).block_ids == [0]


def test_paged_kv_oom_tracks_failure_count() -> None:
    from nano_serve.kv_cache.paged import PagedKVCache

    cache = PagedKVCache(num_blocks=1, block_size=2)
    cache.allocate_prefill("req-a", 2)

    with pytest.raises(MemoryError, match="out of blocks"):
        cache.allocate_decode_slot("req-a")

    assert cache.stats().oom_count == 1


def test_paged_kv_randomized_invariants() -> None:
    import random

    from nano_serve.kv_cache.paged import PagedKVCache

    rng = random.Random(0)
    cache = PagedKVCache(num_blocks=16, block_size=4)
    live: dict[str, int] = {}

    for step in range(100):
        if live and rng.random() < 0.35:
            request_id = rng.choice(list(live))
            cache.free(request_id)
            live.pop(request_id)
            continue

        request_id = f"req-{step}"
        n_tokens = rng.randint(1, 12)
        try:
            cache.allocate_prefill(request_id, n_tokens)
        except MemoryError:
            assert cache.stats().oom_count > 0
            continue
        live[request_id] = n_tokens

        for _ in range(rng.randint(0, 5)):
            try:
                cache.allocate_decode_slot(request_id)
            except MemoryError:
                assert cache.stats().oom_count > 0
                break
            live[request_id] += 1

        stats = cache.stats()
        assert stats.used_tokens == sum(live.values())
        assert stats.used_blocks + stats.free_blocks == stats.num_blocks
        assert 0.0 <= stats.internal_fragmentation <= 1.0
        for live_request_id, seq_len in live.items():
            assert len(cache.get_block_table(live_request_id)) == (
                seq_len + cache.block_size - 1
            ) // cache.block_size


def _make_contiguous_cache(*, max_tokens: int):
    import torch

    from nano_serve.kv_cache.contiguous import ContiguousKVCache

    return ContiguousKVCache(
        num_layers=2,
        max_tokens=max_tokens,
        num_heads=2,
        head_dim=3,
        dtype=torch.float32,
        device="cpu",
    )


def _make_kv(*, num_tokens: int, offset: int = 0):
    import torch

    total = 2 * num_tokens * 2 * 3
    keys = torch.arange(offset, offset + total, dtype=torch.float32).reshape(
        2,
        num_tokens,
        2,
        3,
    )
    values = keys + 1000
    return keys, values
