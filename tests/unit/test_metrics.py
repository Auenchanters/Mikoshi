from __future__ import annotations

import random

import numpy as np
import torch
from conftest import model_config

from aurora.model.transformer import DenseTransformer
from aurora.training.metrics import count_parameters, estimate_transformer_flops
from aurora.training.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


def test_rng_capture_restore_replays_python_numpy_and_torch() -> None:
    seed_everything(9, deterministic=True)
    state = capture_rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(2)

    restore_rng_state(state)

    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    torch.testing.assert_close(torch.rand(2), expected_torch, rtol=0.0, atol=0.0)


def test_parameter_count_does_not_double_count_tied_weights() -> None:
    model = DenseTransformer(model_config(tie_embeddings=True))

    counts = count_parameters(model)

    unique_parameter_total = sum(parameter.numel() for parameter in model.parameters())
    assert counts.total == unique_parameter_total
    assert counts.trainable == unique_parameter_total
    assert counts.active == unique_parameter_total


def test_flop_estimate_matches_hand_calculated_dense_formula() -> None:
    config = model_config()

    forward_flops = estimate_transformer_flops(
        config,
        batch_size=2,
        sequence_length=8,
        training=False,
    )
    training_flops = estimate_transformer_flops(
        config,
        batch_size=2,
        sequence_length=8,
        training=True,
    )

    assert forward_flops == 688_128
    assert training_flops == 2_064_384
