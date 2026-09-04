from __future__ import annotations

import pytest
import torch
from torch import nn

from aurora.model.recurrent_stability import ControlledRecurrentTransformer
from tests.stability_helpers import stability_model_config


def _model(variant: str) -> ControlledRecurrentTransformer:
    if variant == "S1":
        config = stability_model_config(num_iterations=4, input_anchoring=True)
    elif variant == "S2":
        config = stability_model_config(num_iterations=4, gated_update=True)
    elif variant == "S3":
        config = stability_model_config(num_iterations=4, state_stabilization="initial_rms")
    elif variant == "S4":
        config = stability_model_config(num_iterations=4, input_anchoring=True, gated_update=True)
    else:
        raise AssertionError(f"unknown test variant {variant}")
    model = ControlledRecurrentTransformer(config)
    model.cpu()
    return model


def _assert_every_parameter_has_finite_nonzero_gradient(module: nn.Module) -> None:
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    assert parameters
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("variant", ["S1", "S2", "S3", "S4"])
def test_gradients_reach_every_retained_state_and_enabled_control(variant: str) -> None:
    torch.manual_seed(61)
    model = _model(variant)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 64, (2, 8), dtype=torch.long)

    output = model(input_ids, targets=targets, retain_iteration_states=True)
    assert len(output.recurrent_states) == 5
    for state in output.recurrent_states:
        state.retain_grad()
    assert output.loss is not None
    output.loss.backward()

    for state in output.recurrent_states:
        assert state.grad is not None
        assert torch.isfinite(state.grad).all()
        assert float(state.grad.abs().sum()) > 0.0

    _assert_every_parameter_has_finite_nonzero_gradient(model.core)
    if variant in {"S1", "S4"}:
        _assert_every_parameter_has_finite_nonzero_gradient(model.input_anchor)
    if variant in {"S2", "S4"}:
        _assert_every_parameter_has_finite_nonzero_gradient(model.gated_update)
    if variant == "S3":
        assert tuple(model.state_stabilizer.parameters()) == ()
