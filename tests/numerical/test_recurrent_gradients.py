from __future__ import annotations

import torch

from aurora.model.recurrent import FixedLoopRecurrentTransformer
from tests.recurrent_helpers import recurrent_model_config


def test_gradients_flow_through_every_recurrent_state() -> None:
    torch.manual_seed(31)
    model = FixedLoopRecurrentTransformer(recurrent_model_config(num_iterations=4)).cpu()
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
        assert bool(torch.isfinite(state.grad).all())
        assert float(state.grad.abs().sum()) > 0.0
