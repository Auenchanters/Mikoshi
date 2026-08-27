from __future__ import annotations

import torch

from aurora.model.recurrent import FixedLoopRecurrentTransformer
from aurora.model.recurrent_stability import ControlledRecurrentTransformer
from tests.recurrent_helpers import recurrent_model_config
from tests.stability_helpers import stability_model_config


def test_s0_delegates_exactly_to_the_frozen_fixed_loop_model() -> None:
    torch.manual_seed(29)
    base = FixedLoopRecurrentTransformer(recurrent_model_config()).cpu().eval()
    controlled = ControlledRecurrentTransformer(stability_model_config()).cpu().eval()
    controlled.load_state_dict(base.state_dict())
    input_ids = torch.tensor([[1, 4, 7, 3, 9, 2]], dtype=torch.long)
    targets = torch.tensor([[4, 7, 3, 9, 2, 8]], dtype=torch.long)

    assert set(controlled.state_dict()) == set(base.state_dict())
    for name, tensor in base.state_dict().items():
        assert torch.equal(controlled.state_dict()[name], tensor)
    assert sum(parameter.numel() for parameter in controlled.parameters()) == sum(
        parameter.numel() for parameter in base.parameters()
    )

    expected = base(
        input_ids,
        targets=targets,
        collect_diagnostics=True,
        retain_iteration_states=True,
    )
    actual = controlled(
        input_ids,
        targets=targets,
        collect_diagnostics=True,
        retain_iteration_states=True,
    )

    assert actual.gate_diagnostics == ()
    assert actual.iteration_diagnostics == expected.iteration_diagnostics
    assert len(actual.recurrent_states) == len(expected.recurrent_states)
    for actual_state, expected_state in zip(
        actual.recurrent_states, expected.recurrent_states, strict=True
    ):
        assert torch.equal(actual_state, expected_state)
    assert torch.equal(actual.logits, expected.logits)
    assert actual.loss is not None
    assert expected.loss is not None
    assert torch.equal(actual.loss, expected.loss)
