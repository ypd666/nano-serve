"""TileLang simple operator kernels.

This file uses TileLang DSL annotations and intentionally avoids
``from __future__ import annotations``.
"""

from functools import lru_cache

import tilelang
import tilelang.language as T


@tilelang.jit(compile_flags=["-std=c++17"])
def _rmsnorm_kernel(num_rows: int, hidden_size: int, eps: float):
    @T.prim_func
    def main(
        x: T.Tensor((num_rows, hidden_size), "float16"),
        weight: T.Tensor((hidden_size,), "float16"),
        output: T.Tensor((num_rows, hidden_size), "float16"),
    ):
        with T.Kernel(num_rows, T.ceildiv(hidden_size, 128), threads=128) as (row, col_block):
            ss = T.alloc_var("float32", 0.0)
            for d in T.serial(hidden_size):
                value = x[row, d].astype("float32")
                ss += value * value
            inv_rms = T.rsqrt(ss / T.float32(hidden_size) + T.float32(eps))
            for tx in T.Parallel(128):
                col = col_block * 128 + tx
                if col < hidden_size:
                    output[row, col] = (
                        x[row, col].astype("float32")
                        * inv_rms
                        * weight[col].astype("float32")
                    ).astype("float16")

    return main


@tilelang.jit(compile_flags=["-std=c++17"])
def _silu_mul_kernel(num_elements: int):
    @T.prim_func
    def main(
        gate: T.Tensor((num_elements,), "float16"),
        up: T.Tensor((num_elements,), "float16"),
        output: T.Tensor((num_elements,), "float16"),
    ):
        with T.Kernel(T.ceildiv(num_elements, 128), threads=128) as bx:
            for tx in T.Parallel(128):
                idx = bx * 128 + tx
                if idx < num_elements:
                    value = gate[idx].astype("float32")
                    sigmoid = T.float32(1.0) / (T.float32(1.0) + T.exp(-value))
                    output[idx] = (value * sigmoid * up[idx].astype("float32")).astype("float16")

    return main


@tilelang.jit(compile_flags=["-std=c++17"])
def _rope_query_kernel(total_tokens: int, query_heads: int, head_dim: int):
    half_dim = head_dim // 2
    total_q = total_tokens * query_heads * head_dim

    @T.prim_func
    def main(
        q: T.Tensor((total_tokens, query_heads, head_dim), "float16"),
        cos: T.Tensor((total_tokens, head_dim), "float16"),
        sin: T.Tensor((total_tokens, head_dim), "float16"),
        q_out: T.Tensor((total_tokens, query_heads, head_dim), "float16"),
    ):
        with T.Kernel(T.ceildiv(total_q, 128), threads=128) as bx:
            for tx in T.Parallel(128):
                idx = bx * 128 + tx
                if idx < total_q:
                    dim = idx % head_dim
                    head = (idx // head_dim) % query_heads
                    token = idx // (query_heads * head_dim)
                    pair_dim = T.if_then_else(dim < half_dim, dim + half_dim, dim - half_dim)
                    rotate_sign = T.if_then_else(dim < half_dim, T.float32(-1.0), T.float32(1.0))
                    q_out[token, head, dim] = (
                        q[token, head, dim].astype("float32") * cos[token, dim].astype("float32")
                        + rotate_sign
                        * q[token, head, pair_dim].astype("float32")
                        * sin[token, dim].astype("float32")
                    ).astype("float16")

    return main


@tilelang.jit(compile_flags=["-std=c++17"])
def _rope_key_kernel(total_tokens: int, kv_heads: int, head_dim: int):
    half_dim = head_dim // 2
    total_k = total_tokens * kv_heads * head_dim

    @T.prim_func
    def main(
        k: T.Tensor((total_tokens, kv_heads, head_dim), "float16"),
        cos: T.Tensor((total_tokens, head_dim), "float16"),
        sin: T.Tensor((total_tokens, head_dim), "float16"),
        k_out: T.Tensor((total_tokens, kv_heads, head_dim), "float16"),
    ):
        with T.Kernel(T.ceildiv(total_k, 128), threads=128) as bx:
            for tx in T.Parallel(128):
                idx = bx * 128 + tx
                if idx < total_k:
                    dim = idx % head_dim
                    head = (idx // head_dim) % kv_heads
                    token = idx // (kv_heads * head_dim)
                    pair_dim = T.if_then_else(dim < half_dim, dim + half_dim, dim - half_dim)
                    rotate_sign = T.if_then_else(dim < half_dim, T.float32(-1.0), T.float32(1.0))
                    k_out[token, head, dim] = (
                        k[token, head, dim].astype("float32") * cos[token, dim].astype("float32")
                        + rotate_sign
                        * k[token, head, pair_dim].astype("float32")
                        * sin[token, dim].astype("float32")
                    ).astype("float16")

    return main


@tilelang.jit(compile_flags=["-std=c++17"])
def _topk_filter_kernel(num_elements: int, top_k: int):
    @T.prim_func
    def main(
        logits: T.Tensor((num_elements,), "float16"),
        output: T.Tensor((num_elements,), "float16"),
    ):
        with T.Kernel(T.ceildiv(num_elements, 128), threads=128) as bx:
            for tx in T.Parallel(128):
                idx = bx * 128 + tx
                if idx < num_elements:
                    greater_count = T.alloc_var("int32", 0)
                    value = logits[idx].astype("float32")
                    for other in T.serial(num_elements):
                        greater_count += T.if_then_else(
                            logits[other].astype("float32") > value,
                            1,
                            0,
                        )
                    output[idx] = T.if_then_else(
                        greater_count < top_k,
                        logits[idx],
                        T.float32(-65504.0).astype("float16"),
                    )

    return main


@lru_cache(maxsize=32)
def cached_rmsnorm_kernel(num_rows: int, hidden_size: int, eps: float):
    return _rmsnorm_kernel(num_rows, hidden_size, eps)


@lru_cache(maxsize=32)
def cached_silu_mul_kernel(num_elements: int):
    return _silu_mul_kernel(num_elements)


@lru_cache(maxsize=32)
def cached_rope_query_kernel(total_tokens: int, query_heads: int, head_dim: int):
    return _rope_query_kernel(total_tokens, query_heads, head_dim)


@lru_cache(maxsize=32)
def cached_rope_key_kernel(total_tokens: int, kv_heads: int, head_dim: int):
    return _rope_key_kernel(total_tokens, kv_heads, head_dim)


@lru_cache(maxsize=32)
def cached_topk_filter_kernel(num_elements: int, top_k: int):
    return _topk_filter_kernel(num_elements, top_k)
