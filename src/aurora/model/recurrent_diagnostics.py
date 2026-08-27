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
    initial_state_cosine_similarity: float
    relative_update_magnitude: float


def measure_recurrent_state(
    initial: Tensor, previous: Tensor, current: Tensor, *, iteration: int
) -> RecurrentStateDiagnostics:
    if initial.shape != previous.shape or previous.shape != current.shape:
        raise ValueError("initial and successive recurrent states must have identical shapes")
    if previous.ndim < 2:
        raise ValueError("recurrent states must include a batch dimension")
    if iteration <= 0:
        raise ValueError("diagnostic iteration numbers are one-based")
    detached_initial = initial.detach().float()
    detached_previous = previous.detach().float()
    detached_current = current.detach().float()
    update = detached_current - detached_previous
    initial_flat = detached_initial.reshape(detached_initial.shape[0], -1)
    previous_flat = detached_previous.reshape(detached_previous.shape[0], -1)
    current_flat = detached_current.reshape(detached_current.shape[0], -1)
    cosine = functional.cosine_similarity(previous_flat, current_flat, dim=1).mean()
    initial_cosine = functional.cosine_similarity(initial_flat, current_flat, dim=1).mean()
    update_norm = torch.linalg.vector_norm(update)
    current_norm = torch.linalg.vector_norm(detached_current)
    relative_update = update_norm / current_norm
    return RecurrentStateDiagnostics(
        iteration=iteration,
        hidden_state_rms=float(torch.sqrt(detached_current.square().mean()).item()),
        recurrent_update_rms=float(torch.sqrt(update.square().mean()).item()),
        state_cosine_similarity=float(cosine.item()),
        initial_state_cosine_similarity=float(initial_cosine.item()),
        relative_update_magnitude=float(relative_update.item()),
    )
