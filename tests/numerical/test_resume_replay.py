from __future__ import annotations

from pathlib import Path

import torch

from tests.helpers import make_tiny_trainer


def test_cpu_fp32_resume_matches_uninterrupted_training_exactly(tmp_path: Path) -> None:
    uninterrupted = make_tiny_trainer(
        tmp_path / "uninterrupted",
        initialization_seed=17,
        total_steps=8,
    )
    uninterrupted_losses = uninterrupted.train().losses

    split = make_tiny_trainer(
        tmp_path / "split",
        initialization_seed=17,
        total_steps=8,
    )
    first_losses = split.train(max_steps=3).losses
    checkpoint = tmp_path / "resume.pt"
    split.save(checkpoint)

    resumed = make_tiny_trainer(
        tmp_path / "resumed",
        initialization_seed=999,
        total_steps=8,
    )
    resumed.resume(checkpoint)
    second_losses = resumed.train(max_steps=8).losses

    assert first_losses + second_losses == uninterrupted_losses
    assert resumed.step == uninterrupted.step == 8
    assert resumed.tokens_seen == uninterrupted.tokens_seen
    for uninterrupted_parameter, resumed_parameter in zip(
        uninterrupted.model.parameters(),
        resumed.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            uninterrupted_parameter,
            resumed_parameter,
            rtol=0.0,
            atol=0.0,
        )
