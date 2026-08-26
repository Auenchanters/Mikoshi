from __future__ import annotations

import torch

from aurora.model.recurrent import FixedLoopRecurrentTransformer
from aurora.training.reproducibility import seed_everything
from tests.recurrent_helpers import recurrent_model_config


def test_deterministic_cpu_replay_matches_states_logits_loss_and_diagnostics() -> None:
    seed_everything(41, deterministic=True)
    model = FixedLoopRecurrentTransformer(recurrent_model_config(num_iterations=4)).cpu().eval()
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 64, (2, 8), dtype=torch.long)

    first = model(
        input_ids,
        targets=targets,
        collect_diagnostics=True,
        retain_iteration_states=True,
    )
    seed_everything(41, deterministic=True)
    second = model(
        input_ids,
        targets=targets,
        collect_diagnostics=True,
        retain_iteration_states=True,
    )

    assert len(first.recurrent_states) == len(second.recurrent_states) == 5
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.recurrent_states, second.recurrent_states, strict=True)
    )
    assert torch.equal(first.logits, second.logits)
    assert first.loss is not None and second.loss is not None
    assert torch.equal(first.loss, second.loss)
    assert first.iteration_diagnostics == second.iteration_diagnostics


def test_diagnostic_collection_does_not_change_outputs() -> None:
    seed_everything(43, deterministic=True)
    model = FixedLoopRecurrentTransformer(recurrent_model_config(num_iterations=8)).cpu().eval()
    input_ids = torch.randint(0, 64, (1, 8), dtype=torch.long)
    targets = torch.randint(0, 64, (1, 8), dtype=torch.long)

    plain = model(input_ids, targets=targets)
    observed = model(input_ids, targets=targets, collect_diagnostics=True)

    assert plain.iteration_diagnostics == ()
    assert len(observed.iteration_diagnostics) == 8
    assert torch.equal(plain.logits, observed.logits)
    assert plain.loss is not None and observed.loss is not None
    assert torch.equal(plain.loss, observed.loss)
