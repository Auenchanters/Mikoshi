from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True)
class RecurrentStateDiagnostics:
    iteration: int
    hidden_state_rms: float
    recurrent_update_rms: float
    state_cosine_similarity: float


def measure_recurrent_state(
    previous: Tensor, current: Tensor, *, iteration: int
) -> RecurrentStateDiagnostics:
    if previous.shape != current.shape:
        raise ValueError("successive recurrent states must have identical shapes")
    if previous.ndim < 2:
        raise ValueError("recurrent states must include a batch dimension")
    if iteration <= 0:
        raise ValueError("diagnostic iteration numbers are one-based")
    detached_previous = previous.detach().float()
    detached_current = current.detach().float()
    update = detached_current - detached_previous
    previous_flat = detached_previous.reshape(detached_previous.shape[0], -1)
    current_flat = detached_current.reshape(detached_current.shape[0], -1)
    cosine = functional.cosine_similarity(previous_flat, current_flat, dim=1).mean()
    return RecurrentStateDiagnostics(
        iteration=iteration,
        hidden_state_rms=float(torch.sqrt(detached_current.square().mean()).item()),
        recurrent_update_rms=float(torch.sqrt(update.square().mean()).item()),
        state_cosine_similarity=float(cosine.item()),
    )
