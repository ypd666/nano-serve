"""TileLang paged decode attention kernels.

This module intentionally avoids ``from __future__ import annotations`` because
TileLang evaluates tensor annotations while building TIR.
"""

from functools import lru_cache

import tilelang
import tilelang.language as T


@tilelang.jit(compile_flags=["-std=c++17"])
def _paged_decode_kernel(
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    context_len: int,
    block_size: int,
    max_blocks: int,
    num_blocks: int,
):
    scale = head_dim**-0.5
    total = batch_size * query_heads * head_dim
    head_group = query_heads // kv_heads

    @T.prim_func
    def main(
        query: T.Tensor((batch_size, query_heads, 1, head_dim), "float16"),
        paged_key: T.Tensor((num_blocks, kv_heads, block_size, head_dim), "float16"),
        paged_value: T.Tensor((num_blocks, kv_heads, block_size, head_dim), "float16"),
        block_tables: T.Tensor((batch_size, max_blocks), "int32"),
        seq_lens: T.Tensor((batch_size,), "int32"),
        output: T.Tensor((batch_size, query_heads, 1, head_dim), "float16"),
    ):
        with T.Kernel(T.ceildiv(total, 128), threads=128) as bx:
            for tx in T.Parallel(128):
                idx = bx * 128 + tx
                if idx < total:
                    dim = idx % head_dim
                    head = (idx // head_dim) % query_heads
                    batch = idx // (query_heads * head_dim)
                    kv_head = head // head_group
                    seq_len = seq_lens[batch]
                    max_score = T.alloc_var("float32", -3.4028234663852886e38)

                    for pos in T.serial(context_len):
                        if pos < seq_len:
                            block_slot = pos // block_size
                            block_offset = pos - block_slot * block_size
                            block_id = block_tables[batch, block_slot]
                            score = T.alloc_var("float32", 0.0)
                            for d in T.serial(head_dim):
                                score += (
                                    query[batch, head, 0, d].astype("float32")
                                    * paged_key[
                                        block_id,
                                        kv_head,
                                        block_offset,
                                        d,
                                    ].astype("float32")
                                )
                            score *= T.float32(scale)
                            if score > max_score:
                                max_score = score

                    denom = T.alloc_var("float32", 0.0)
                    acc = T.alloc_var("float32", 0.0)
                    for pos in T.serial(context_len):
                        if pos < seq_len:
                            block_slot = pos // block_size
                            block_offset = pos - block_slot * block_size
                            block_id = block_tables[batch, block_slot]
                            score = T.alloc_var("float32", 0.0)
                            for d in T.serial(head_dim):
                                score += (
                                    query[batch, head, 0, d].astype("float32")
                                    * paged_key[
                                        block_id,
                                        kv_head,
                                        block_offset,
                                        d,
                                    ].astype("float32")
                                )
                            score *= T.float32(scale)
                            weight = T.exp(score - max_score)
                            denom += weight
                            acc += (
                                weight
                                * paged_value[
                                    block_id,
                                    kv_head,
                                    block_offset,
                                    dim,
                                ].astype("float32")
                            )
                    output[batch, head, 0, dim] = (acc / denom).astype("float16")

    return main


@lru_cache(maxsize=32)
def cached_paged_decode_kernel(
    batch_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    context_len: int,
    block_size: int,
    max_blocks: int,
    num_blocks: int,
):
    return _paged_decode_kernel(
        batch_size,
        query_heads,
        kv_heads,
        head_dim,
        context_len,
        block_size,
        max_blocks,
        num_blocks,
    )
