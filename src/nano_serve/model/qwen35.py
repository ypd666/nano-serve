"""Text-only Qwen3.5 torch modules for Phase 1.

This is a narrow full-context, no-cache implementation for
`Qwen/Qwen3.5-4B`. It intentionally does not include vision inputs, KV cache,
batch scheduling, or generation policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Qwen35TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    head_dim: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    partial_rotary_factor: float
    layer_types: tuple[str, ...]
    pad_token_id: int | None = None
    eos_token_id: int | None = None

    @classmethod
    def from_model_config(cls, config: dict[str, Any]) -> "Qwen35TextConfig":
        text_config = config.get("text_config", config)
        if not isinstance(text_config, dict):
            raise ValueError("Qwen3.5 text_config must be a JSON object")
        if text_config.get("model_type") not in {"qwen3_5_text", None}:
            raise ValueError(f"Unsupported text model type: {text_config.get('model_type')}")

        rope_parameters = text_config.get("rope_parameters") or {}
        layer_types = tuple(str(item) for item in text_config["layer_types"])
        if len(layer_types) != int(text_config["num_hidden_layers"]):
            raise ValueError("layer_types length must match num_hidden_layers")
        unsupported = sorted(set(layer_types) - {"linear_attention", "full_attention"})
        if unsupported:
            raise ValueError(f"Unsupported Qwen3.5 layer types: {unsupported}")

        return cls(
            vocab_size=int(text_config["vocab_size"]),
            hidden_size=int(text_config["hidden_size"]),
            intermediate_size=int(text_config["intermediate_size"]),
            num_hidden_layers=int(text_config["num_hidden_layers"]),
            num_attention_heads=int(text_config["num_attention_heads"]),
            num_key_value_heads=int(text_config["num_key_value_heads"]),
            rms_norm_eps=float(text_config["rms_norm_eps"]),
            hidden_act=str(text_config["hidden_act"]),
            attention_bias=bool(text_config.get("attention_bias", False)),
            attention_dropout=float(text_config.get("attention_dropout", 0.0)),
            head_dim=int(text_config["head_dim"]),
            linear_conv_kernel_dim=int(text_config["linear_conv_kernel_dim"]),
            linear_key_head_dim=int(text_config["linear_key_head_dim"]),
            linear_value_head_dim=int(text_config["linear_value_head_dim"]),
            linear_num_key_heads=int(text_config["linear_num_key_heads"]),
            linear_num_value_heads=int(text_config["linear_num_value_heads"]),
            max_position_embeddings=int(text_config["max_position_embeddings"]),
            rope_theta=float(rope_parameters.get("rope_theta", 10000.0)),
            partial_rotary_factor=float(rope_parameters.get("partial_rotary_factor", 1.0)),
            layer_types=layer_types,
            pad_token_id=_optional_int(text_config.get("pad_token_id")),
            eos_token_id=_optional_int(text_config.get("eos_token_id")),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, list):
        return int(value[0]) if value else None
    return int(value)


class Qwen35RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen35RMSNormGated(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        output = hidden_states.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = self.weight * output.to(input_dtype)
        output = output * F.silu(gate.float())
        return output.to(input_dtype)


class Qwen35TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen35TextConfig, *, device: torch.device) -> None:
        super().__init__()
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
                / rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x.shape[:2]
        position_ids = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        inv_freq = cast(torch.Tensor, self.inv_freq)
        freqs = torch.outer(position_ids, inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return (
            cos.unsqueeze(0).expand(batch_size, -1, -1),
            sin.unsqueeze(0).expand(batch_size, -1, -1),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        repeats,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * repeats, seq_len, head_dim)


def _l2norm(x: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        item.transpose(1, 2).contiguous().to(torch.float32)
        for item in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        item.reshape(item.shape[0], item.shape[1], -1, chunk_size, item.shape[-1])
        for item in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    full_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(full_mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    attn = attn + eye
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    recurrent_state = torch.zeros(
        batch_size,
        num_heads,
        k_head_dim,
        v_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    core_attn_out = torch.zeros_like(value)
    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        chunk_attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ recurrent_state
        core_attn_out[:, :, i] = attn_inter + chunk_attn @ v_new
        recurrent_state = (
            recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1,
                -2,
            )
            @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0],
        core_attn_out.shape[1],
        -1,
        core_attn_out.shape[-1],
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        layer_idx: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, device=device, dtype=dtype))
        a_log = torch.empty(self.num_v_heads, device=device, dtype=torch.float32)
        self.A_log = nn.Parameter(a_log.uniform_(0, 16).log_().to(dtype))
        self.norm = Qwen35RMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            self.value_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.in_proj_qkv = nn.Linear(
            config.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.in_proj_z = nn.Linear(
            config.hidden_size,
            self.value_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.in_proj_b = nn.Linear(
            config.hidden_size,
            self.num_v_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.in_proj_a = nn.Linear(
            config.hidden_size,
            self.num_v_heads,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size,
            seq_len,
            -1,
            self.head_v_dim,
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        core_attn_out = _torch_chunk_gated_delta_rule(query, key, value, g, beta)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class Qwen35Attention(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        layer_idx: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
            device=device,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            device=device,
            dtype=dtype,
        )
        self.q_norm = Qwen35RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.k_norm = Qwen35RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        seq_len = hidden_states.shape[1]
        causal_mask = _causal_mask(seq_len, device=hidden_states.device, dtype=attn_weights.dtype)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


def _causal_mask(seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full(
        (seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)


class Qwen35MLP(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported Qwen3.5 activation: {config.hidden_act}")
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        layer_idx: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(
                config,
                layer_idx,
                device=device,
                dtype=dtype,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(
                config,
                layer_idx,
                device=device,
                dtype=dtype,
            )
        self.mlp = Qwen35MLP(config, device=device, dtype=dtype)
        self.input_layernorm = Qwen35RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen35TextModel(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                Qwen35DecoderLayer(config, layer_idx, device=device, dtype=dtype)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen35RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.rotary_emb = Qwen35TextRotaryEmbedding(config, device=device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)
        return self.norm(hidden_states)


class Qwen35ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen35TextModel(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return self.model.embed_tokens.weight.device

    def forward(self, input_ids: torch.Tensor, *, logits_to_keep: int = 0) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        return self.lm_head(hidden_states)

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids, logits_to_keep=1)[:, -1, :]
