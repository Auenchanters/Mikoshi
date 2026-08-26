from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with fp32 reduction math."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, inputs: Tensor) -> Tensor:
        input_dtype = inputs.dtype
        values = inputs.float()
        inverse_rms = torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = values * inverse_rms
        return (normalized * self.weight.float()).to(dtype=input_dtype)
