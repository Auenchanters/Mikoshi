from __future__ import annotations

import torch
from torch import nn

from aurora.model.recurrent import FixedLoopRecurrentTransformer
from aurora.training.metrics import count_parameters
from tests.recurrent_helpers import recurrent_model_config


def test_every_iteration_invokes_the_same_core_object_and_parameters() -> None:
    torch.manual_seed(11)
    model = FixedLoopRecurrentTransformer(recurrent_model_config(num_iterations=4)).eval()
    calls: list[nn.Module] = []
    parameter_locations = {
        name: (id(parameter), parameter.data_ptr(), parameter.untyped_storage().data_ptr())
        for name, parameter in model.core.named_parameters()
    }
    handle = model.core.register_forward_hook(lambda module, _inputs, _output: calls.append(module))

    try:
        model(torch.randint(0, 64, (2, 8), dtype=torch.long))
    finally:
        handle.remove()

    assert len(calls) == 4
    assert all(module is model.core for module in calls)
    assert parameter_locations == {
        name: (id(parameter), parameter.data_ptr(), parameter.untyped_storage().data_ptr())
        for name, parameter in model.core.named_parameters()
    }


def test_parameter_count_and_state_keys_are_invariant_across_iteration_counts() -> None:
    counts: list[tuple[int, int, int]] = []
    key_sets: list[set[str]] = []
    for num_iterations in (1, 2, 4, 8):
        model = FixedLoopRecurrentTransformer(recurrent_model_config(num_iterations=num_iterations))
        parameter_counts = count_parameters(model)
        counts.append(
            (
                parameter_counts.total,
                parameter_counts.trainable,
                parameter_counts.active,
            )
        )
        key_sets.append(set(model.state_dict()))

    assert len(set(counts)) == 1
    assert all(keys == key_sets[0] for keys in key_sets)
    core_keys = {key for key in key_sets[0] if key.startswith("core.")}
    assert core_keys
    assert not any("iteration" in key for key in core_keys)


def test_recurrent_forward_produces_aligned_logits_and_loss() -> None:
    model = FixedLoopRecurrentTransformer(recurrent_model_config(num_iterations=2))
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 64, (2, 8), dtype=torch.long)

    output = model(input_ids, targets=targets)

    assert output.logits.shape == (2, 8, 64)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
