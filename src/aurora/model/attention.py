from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from aurora.config import ModelConfig
from aurora.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with fewer key/value heads than query heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("attention head_dim must be even for RoPE")
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.dropout = config.dropout
        self.qk_norm = config.qk_norm
        self.norm_eps = config.norm_eps
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

    def _normalize_qk(self, values: Tensor) -> Tensor:
        input_dtype = values.dtype
        fp32_values = values.float()
        inverse_rms = torch.rsqrt(fp32_values.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
        return (fp32_values * inverse_rms).to(dtype=input_dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("attention expects [batch, sequence, d_model] inputs")
        batch_size, sequence_length, _ = inputs.shape
        query = self.q_proj(inputs).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.k_proj(inputs).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(inputs).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rope(query, key)
        if self.qk_norm:
            query = self._normalize_qk(query)
            key = self._normalize_qk(key)
        repeat_factor = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)
        dropout_probability = self.dropout if self.training else 0.0
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_probability,
            is_causal=True,
        )
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        )
        return self.out_proj(attended)
