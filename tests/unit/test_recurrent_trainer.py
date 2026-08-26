from __future__ import annotations

import json
from pathlib import Path

import torch

from aurora.data.dataset import TokenBlockDataset
from aurora.experiment import create_run
from aurora.model.recurrent import FixedLoopRecurrentTransformer
from aurora.training.metrics import estimate_recurrent_transformer_flops
from aurora.training.recurrent_trainer import RecurrentTrainer
from aurora.training.reproducibility import seed_everything
from tests.fixtures.byte_tokenizer import ByteTokenizer
from tests.recurrent_helpers import recurrent_experiment_config, recurrent_model_config


def _make_recurrent_trainer(root: Path) -> RecurrentTrainer:
    config = recurrent_experiment_config(num_iterations=2, total_steps=2)
    seed_everything(config.runtime.seed, deterministic=True)
    model = FixedLoopRecurrentTransformer(config.model)
    pattern = [4, 5, 6, 7, 8, 7, 6, 5]
    dataset = TokenBlockDataset(
        pattern * 32,
        sequence_length=config.data.sequence_length,
        stride=config.data.stride,
    )
    run = create_run(config, root / "runs")
    return RecurrentTrainer(
        config=config,
        model=model,
        tokenizer_identity=ByteTokenizer().identity,
        train_dataset=dataset,
        eval_dataset=dataset,
        run=run,
    )


def test_recurrent_flops_use_active_core_calls_not_unique_core_count() -> None:
    config = recurrent_model_config(num_iterations=2)

    forward_flops = estimate_recurrent_transformer_flops(
        config,
        batch_size=2,
        sequence_length=8,
        training=False,
    )
    training_flops = estimate_recurrent_transformer_flops(
        config,
        batch_size=2,
        sequence_length=8,
        training=True,
    )

    assert forward_flops == 1_310_720
    assert training_flops == 3_932_160


def test_optimizer_contains_each_shared_parameter_exactly_once(tmp_path: Path) -> None:
    trainer = _make_recurrent_trainer(tmp_path)
    model_parameter_ids = {
        id(parameter) for parameter in trainer.model.parameters() if parameter.requires_grad
    }
    optimizer_parameter_ids = [
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    ]

    assert len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids))
    assert set(optimizer_parameter_ids) == model_parameter_ids


def test_recurrent_trainer_logs_state_and_core_gradient_diagnostics(tmp_path: Path) -> None:
    trainer = _make_recurrent_trainer(tmp_path)

    result = trainer.train(max_steps=1)

    records = [
        json.loads(line)
        for line in trainer.run.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.steps_completed == 1
    assert len(records) == 1
    record = records[0]
    assert record["recurrent_iterations"] == 2
    assert len(record["recurrent_diagnostics"]) == 2
    assert record["gradient_norm"] > 0.0
    assert record["core_gradient_l2_norm"] > 0.0
    assert record["core_gradient_finite"] is True
    assert record["core_gradient_nonzero_fraction"] > 0.0
    assert torch.isfinite(torch.tensor(record["estimated_training_flops"]))
