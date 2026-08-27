from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers import make_tiny_trainer


def test_trainer_reduces_loss_on_repeated_pattern(tmp_path: Path) -> None:
    trainer = make_tiny_trainer(tmp_path, total_steps=30)

    result = trainer.train()

    assert result.steps_completed == 30
    assert result.final_loss < result.initial_loss
    evaluation = trainer.evaluate(max_batches=4)
    assert evaluation.loss >= 0.0
    assert evaluation.target_tokens == 4 * 4 * 8
    metric_records = [
        json.loads(line)
        for line in trainer.run.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert metric_records[-1]["step"] == 30
    assert metric_records[-1]["gradient_norm"] >= 0.0
    assert metric_records[-1]["learning_rate"] >= 0.0
    assert metric_records[-1]["tokens_per_second"] > 0.0
    metadata = json.loads(trainer.run.metadata_path.read_text(encoding="utf-8"))
    assert metadata["numpy_version"] == np.__version__
    assert metadata["torch_version"] == torch.__version__
    assert metadata["git_commit"]
    assert metadata["hardware"]


def test_trainer_stops_on_non_finite_loss(tmp_path: Path) -> None:
    trainer = make_tiny_trainer(tmp_path, total_steps=1)
    with torch.no_grad():
        next(trainer.model.parameters()).fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        trainer.train()
