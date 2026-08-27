from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        return self.dropout(self.down_proj(hidden))
