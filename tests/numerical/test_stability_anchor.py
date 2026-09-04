from __future__ import annotations

import torch

from aurora.model.recurrent_stability import ControlledRecurrentTransformer, InputAnchor
from tests.stability_helpers import stability_model_config


def test_input_anchor_projects_initial_state_and_adds_recurrent_state() -> None:
    anchor = InputAnchor(2)
    with torch.no_grad():
        anchor.projection.weight.copy_(torch.tensor([[2.0, -1.0], [0.5, 3.0]]))
    initial_state = torch.tensor([[[1.0, 2.0]]])
    recurrent_state = torch.tensor([[[4.0, 5.0]]])

    anchored = anchor(initial_state, recurrent_state)

    expected = recurrent_state + anchor.projection(initial_state)
    assert torch.equal(anchored, expected)


def test_model_anchors_every_iteration_to_the_original_prelude_output() -> None:
    model = ControlledRecurrentTransformer(
        stability_model_config(num_iterations=4, input_anchoring=True)
    ).eval()
    input_ids = torch.tensor([[1, 4, 7, 3, 9, 2]], dtype=torch.long)
    prelude_outputs: list[torch.Tensor] = []
    anchor_initial_states: list[torch.Tensor] = []
    anchor_recurrent_states: list[torch.Tensor] = []

    def record_prelude(
        _module: object, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        prelude_outputs.append(output)

    def record_anchor(_module: object, inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
        anchor_initial_states.append(inputs[0])
        anchor_recurrent_states.append(inputs[1])

    prelude_hook = model.prelude.register_forward_hook(record_prelude)
    anchor_hook = model.input_anchor.register_forward_pre_hook(record_anchor)
    try:
        model(input_ids, retain_iteration_states=True)
    finally:
        prelude_hook.remove()
        anchor_hook.remove()

    assert len(prelude_outputs) == 1
    assert len(anchor_initial_states) == 4
    initial_state = prelude_outputs[0]
    assert all(state is initial_state for state in anchor_initial_states)
    assert all(state is not initial_state for state in anchor_recurrent_states[1:])
