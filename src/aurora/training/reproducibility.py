from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int, deterministic: bool) -> None:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def capture_rng_state() -> dict[str, object]:
    return {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    required = {"numpy", "python", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("RNG state has missing or unknown keys")
    random.setstate(cast(tuple[Any, ...], state["python"]))
    np.random.set_state(cast(tuple[str, np.ndarray, int, int, float], state["numpy"]))
    torch.set_rng_state(cast(Tensor, state["torch_cpu"]))
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cast(list[Tensor], cuda_state))
