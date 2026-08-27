from __future__ import annotations

import pytest
import torch

from aurora.model.recurrent_stability import (
    ControlledRecurrentTransformer,
    GatedRecurrentUpdate,
    RecurrentGateDiagnostics,
)
from tests.stability_helpers import stability_model_config


def test_gate_interpolates_previous_and_proposal_from_scalar_sigmoid() -> None:
    update = GatedRecurrentUpdate(d_model=2)
    with torch.no_grad():
        update.projection.weight.copy_(torch.tensor([[0.5, -1.0, 2.0, 0.25]]))
        update.projection.bias.copy_(torch.tensor([-0.75]))
    previous = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    proposal = torch.tensor([[[2.0, 1.0], [-1.0, 4.0]]])

    state, gate = update(previous, proposal)

    expected_gate = torch.tensor([[[0.997_527_36], [0.010_986_94]]])
    assert torch.allclose(gate, expected_gate, atol=1e-7, rtol=0.0)
    assert torch.isfinite(gate).all()
    assert ((gate >= 0.0) & (gate <= 1.0)).all()
    expected_state = (1.0 - gate) * previous + gate * proposal
    assert torch.allclose(state, expected_state, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize(
    ("bias", "expected_label"),
    [(-3.0, "zero"), (3.0, "one"), (0.0, "none")],
)
def test_s2_records_detached_gate_statistics_and_collapse_label(
    bias: float, expected_label: str
) -> None:
    model = ControlledRecurrentTransformer(
        stability_model_config(num_iterations=2, gated_update=True)
    ).eval()
    with torch.no_grad():
        model.gated_update.projection.weight.zero_()
        model.gated_update.projection.bias.fill_(bias)

    output = model(torch.tensor([[1, 4, 7, 3]], dtype=torch.long), collect_diagnostics=True)

    expected_gate = torch.sigmoid(torch.tensor(bias)).item()
    assert len(output.gate_diagnostics) == 2
    for iteration, diagnostic in enumerate(output.gate_diagnostics, start=1):
        assert isinstance(diagnostic, RecurrentGateDiagnostics)
        assert diagnostic.iteration == iteration
        assert diagnostic.mean == expected_gate
        assert diagnostic.population_std == 0.0
        assert diagnostic.minimum == expected_gate
        assert diagnostic.maximum == expected_gate
        assert diagnostic.collapse == expected_label
