from __future__ import annotations

from nano_serve.attention import TorchGatherPagedAttention


def test_paged_gather_reconstructs_block_boundary_context() -> None:
    import torch

    attention = TorchGatherPagedAttention()
    key = torch.arange(3 * 2 * 4 * 3, dtype=torch.float32).reshape(3, 2, 4, 3)
    value = key + 1000

    gathered_key, gathered_value = attention.gather_kv(
        key,
        value,
        block_tables=[[2, 0]],
        seq_lens=[6],
    )

    expected_key = torch.cat((key[2, :, :4], key[0, :, :2]), dim=1).unsqueeze(0)
    expected_value = torch.cat((value[2, :, :4], value[0, :, :2]), dim=1).unsqueeze(0)

    torch.testing.assert_close(gathered_key, expected_key)
    torch.testing.assert_close(gathered_value, expected_value)


def test_paged_decode_matches_contiguous_attention() -> None:
    import torch

    torch.manual_seed(0)
    attention = TorchGatherPagedAttention()
    query = torch.randn(2, 4, 1, 8)
    contiguous_key = torch.randn(2, 2, 5, 8)
    contiguous_value = torch.randn(2, 2, 5, 8)
    paged_key, paged_value, block_tables = _pack_contiguous(contiguous_key, contiguous_value, 3)

    actual, metadata = attention.forward_decode(
        query,
        paged_key,
        paged_value,
        block_tables,
        seq_lens=[5, 5],
    )
    expected, _ = attention.forward_prefill(
        query,
        contiguous_key,
        contiguous_value,
        causal=False,
    )

    torch.testing.assert_close(actual, expected)
    assert metadata.context_tokens == 5
    assert metadata.block_size == 3
    assert metadata.gather_time_ms >= 0


def test_paged_decode_supports_mqa() -> None:
    import torch

    torch.manual_seed(1)
    attention = TorchGatherPagedAttention()
    query = torch.randn(1, 4, 1, 4)
    contiguous_key = torch.randn(1, 1, 7, 4)
    contiguous_value = torch.randn(1, 1, 7, 4)
    paged_key, paged_value, block_tables = _pack_contiguous(contiguous_key, contiguous_value, 4)

    actual, _ = attention.forward_decode(
        query,
        paged_key,
        paged_value,
        block_tables,
        seq_lens=[7],
    )
    expected, _ = attention.forward_prefill(
        query,
        contiguous_key,
        contiguous_value,
        causal=False,
    )

    torch.testing.assert_close(actual, expected)


def test_paged_decode_masks_mixed_sequence_lengths() -> None:
    import torch

    torch.manual_seed(3)
    attention = TorchGatherPagedAttention()
    query = torch.randn(2, 2, 1, 4)
    first_key = torch.randn(1, 1, 5, 4)
    first_value = torch.randn(1, 1, 5, 4)
    second_key = torch.randn(1, 1, 3, 4)
    second_value = torch.randn(1, 1, 3, 4)
    paged_key = torch.zeros(3, 1, 4, 4)
    paged_value = torch.zeros_like(paged_key)
    paged_key[0, :, :4] = first_key[0, :, :4]
    paged_value[0, :, :4] = first_value[0, :, :4]
    paged_key[1, :, :1] = first_key[0, :, 4:5]
    paged_value[1, :, :1] = first_value[0, :, 4:5]
    paged_key[2, :, :3] = second_key[0]
    paged_value[2, :, :3] = second_value[0]

    actual, _ = attention.forward_decode(
        query,
        paged_key,
        paged_value,
        block_tables=[[0, 1], [2]],
        seq_lens=[5, 3],
    )
    first_expected, _ = attention.forward_prefill(
        query[:1],
        first_key,
        first_value,
        causal=False,
    )
    second_expected, _ = attention.forward_prefill(
        query[1:],
        second_key,
        second_value,
        causal=False,
    )

    torch.testing.assert_close(actual, torch.cat((first_expected, second_expected), dim=0))


def test_paged_prefill_causal_matches_manual_attention() -> None:
    import torch

    torch.manual_seed(2)
    attention = TorchGatherPagedAttention()
    query = torch.randn(1, 2, 4, 3)
    key = torch.randn(1, 2, 4, 3)
    value = torch.randn(1, 2, 4, 3)

    actual, _ = attention.forward_prefill(query, key, value, causal=True)
    expected = []
    for token_index in range(4):
        token_output, _ = attention.forward_prefill(
            query[:, :, token_index : token_index + 1],
            key[:, :, : token_index + 1],
            value[:, :, : token_index + 1],
            causal=False,
        )
        expected.append(token_output)

    torch.testing.assert_close(actual, torch.cat(expected, dim=2))


def _pack_contiguous(contiguous_key, contiguous_value, block_size: int):
    import torch

    batch, heads, seq_len, head_dim = contiguous_key.shape
    block_tables: list[list[int]] = []
    key_blocks = []
    value_blocks = []
    for batch_index in range(batch):
        block_ids = []
        for start in range(0, seq_len, block_size):
            block_id = len(key_blocks)
            block_ids.append(block_id)
            key_block = torch.zeros((heads, block_size, head_dim), dtype=contiguous_key.dtype)
            value_block = torch.zeros_like(key_block)
            end = min(seq_len, start + block_size)
            block_tokens = end - start
            key_block[:, :block_tokens] = contiguous_key[batch_index, :, start:end]
            value_block[:, :block_tokens] = contiguous_value[batch_index, :, start:end]
            key_blocks.append(key_block)
            value_blocks.append(value_block)
        block_tables.append(block_ids)
    return torch.stack(key_blocks), torch.stack(value_blocks), block_tables
