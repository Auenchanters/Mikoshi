from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Interleaved rotary position embedding with a fixed reference cache."""

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if base <= 0.0:
            raise ValueError("base must be positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos_cached", frequencies.cos(), persistent=False)
        self.register_buffer("sin_cached", frequencies.sin(), persistent=False)

    @staticmethod
    def _rotate(inputs: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        input_dtype = inputs.dtype
        values = inputs.float()
        even = values[..., 0::2]
        odd = values[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(-2)
        return rotated.to(dtype=input_dtype)

    def forward(self, query: Tensor, key: Tensor, start_pos: int = 0) -> tuple[Tensor, Tensor]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("RoPE expects [batch, heads, sequence, head_dim] tensors")
        if query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError("query/key head dimension does not match RoPE")
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("query and key sequence lengths must match")
        sequence_length = query.shape[-2]
        if start_pos < 0 or start_pos + sequence_length > self.max_seq_len:
            raise ValueError("rotary positions exceed max_seq_len")
        cosine = self.cos_cached[start_pos : start_pos + sequence_length].to(device=query.device)
        sine = self.sin_cached[start_pos : start_pos + sequence_length].to(device=query.device)
        cosine = cosine.view(1, 1, sequence_length, -1)
        sine = sine.view(1, 1, sequence_length, -1)
        return self._rotate(query, cosine, sine), self._rotate(key, cosine, sine)
