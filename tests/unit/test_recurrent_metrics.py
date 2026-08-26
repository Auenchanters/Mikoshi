from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from aurora.model.recurrent_diagnostics import measure_recurrent_state
from aurora.training.recurrent_metrics import collect_core_gradient_statistics


def test_recurrent_state_statistics_match_hand_derived_values() -> None:
    previous = torch.tensor([[[1.0, 0.0]]])
    current = torch.tensor([[[0.0, 2.0]]])

    measured = measure_recurrent_state(previous, current, iteration=1)

    assert measured.iteration == 1
    assert measured.hidden_state_rms == pytest.approx(math.sqrt(2.0))
    assert measured.recurrent_update_rms == pytest.approx(math.sqrt(2.5))
    assert measured.state_cosine_similarity == pytest.approx(0.0)


def test_core_gradient_statistics_match_literal_gradient() -> None:
    core = nn.Linear(2, 1, bias=False)
    core.weight.grad = torch.tensor([[3.0, 4.0]])

    measured = collect_core_gradient_statistics(core)

    assert measured.l2_norm == pytest.approx(5.0)
    assert measured.mean_abs == pytest.approx(3.5)
    assert measured.max_abs == pytest.approx(4.0)
    assert measured.nonzero_fraction == pytest.approx(1.0)
    assert measured.finite is True
    assert measured.parameters_with_grad == measured.trainable_parameters == 2
