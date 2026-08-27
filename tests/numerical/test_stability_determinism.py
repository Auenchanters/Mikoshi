from __future__ import annotations

import torch

from aurora.model.recurrent_stability import ControlledRecurrentTransformer
from aurora.training.reproducibility import seed_everything
from tests.stability_helpers import stability_model_config


def test_s2_cpu_replay_exactly_matches_gate_diagnostics() -> None:
    seed_everything(47, deterministic=True)
    model = ControlledRecurrentTransformer(
        stability_model_config(num_iterations=4, gated_update=True)
    ).cpu().eval()
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 64, (2, 8), dtype=torch.long)

    first = model(
        input_ids,
        targets=targets,
        collect_diagnostics=True,
        retain_iteration_states=True,
    )
    seed_everything(47, deterministic=True)
    second = model(
        input_ids,
        targets=targets,
        collect_diagnostics=True,
        retain_iteration_states=True,
    )

    assert first.gate_diagnostics == second.gate_diagnostics
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.recurrent_states, second.recurrent_states, strict=True)
    )
    assert torch.equal(first.logits, second.logits)
    assert first.loss is not None and second.loss is not None
    assert torch.equal(first.loss, second.loss)


def test_s2_gate_diagnostics_do_not_change_outputs() -> None:
    seed_everything(53, deterministic=True)
    model = ControlledRecurrentTransformer(
        stability_model_config(num_iterations=4, gated_update=True)
    ).cpu().eval()
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 64, (2, 8), dtype=torch.long)

    plain = model(input_ids, targets=targets, retain_iteration_states=True)
    observed = model(
        input_ids,
        targets=targets,
        collect_diagnostics=True,
        retain_iteration_states=True,
    )

    assert plain.gate_diagnostics == ()
    assert len(observed.gate_diagnostics) == 4
    assert all(
        torch.equal(left, right)
        for left, right in zip(plain.recurrent_states, observed.recurrent_states, strict=True)
    )
    assert torch.equal(plain.logits, observed.logits)
    assert plain.loss is not None and observed.loss is not None
    assert torch.equal(plain.loss, observed.loss)
