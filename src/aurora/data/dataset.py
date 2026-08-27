from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose shifted causal-LM windows from one deterministic token stream."""

    def __init__(
        self,
        tokens: Sequence[int] | Tensor,
        sequence_length: int,
        stride: int | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        resolved_stride = sequence_length if stride is None else stride
        if resolved_stride <= 0:
            raise ValueError("stride must be positive")
        if isinstance(tokens, Tensor):
            tensor = tokens.detach().to(device="cpu", dtype=torch.long).clone()
        else:
            tensor = torch.tensor(list(tokens), dtype=torch.long)
        if tensor.ndim != 1:
            raise ValueError("tokens must be a one-dimensional stream")
        token_count = cast(int, tensor.numel())
        if token_count < sequence_length + 1:
            raise ValueError("token stream must contain at least sequence_length + 1 tokens")
        self._tokens = tensor
        self.sequence_length = sequence_length
        self.stride = resolved_stride
        self._length = 1 + (token_count - sequence_length - 1) // resolved_stride

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if not 0 <= index < self._length:
            raise IndexError(index)
        start = index * self.stride
        stop = start + self.sequence_length
        return self._tokens[start:stop], self._tokens[start + 1 : stop + 1]
