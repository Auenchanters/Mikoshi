from __future__ import annotations

import json
from pathlib import Path

from aurora.data.dataset import TokenBlockDataset
from aurora.experiment import create_run
from aurora.model.recurrent_stability import ControlledRecurrentTransformer
from aurora.stability_config import StabilityExperimentConfig
from aurora.training.metrics import count_parameters
from aurora.training.reproducibility import seed_everything
from aurora.training.stability_trainer import StabilityTrainer
from tests.fixtures.byte_tokenizer import ByteTokenizer
from tests.stability_helpers import stability_experiment_payload


def _config(*, num_iterations: int = 2, total_steps: int = 2) -> StabilityExperimentConfig:
    payload = stability_experiment_payload(
        num_iterations=num_iterations,
        gated_update=True,
        max_seq_len=8,
        d_model=24,
        d_ff=48,
    )
    payload["experiment"] = {
        "run_id": "EXP-9602-stability-trainer-test",
        "description": "Stability trainer test",
        "dataset_version": "test-pattern-v1",
    }
    payload["data"] = {"sequence_length": 8, "batch_size": 2, "stride": 4}
    payload["scheduler"] = {
        "warmup_steps": 0,
        "total_steps": total_steps,
        "min_lr_ratio": 0.2,
    }
    payload["runtime"] = {
        "seed": 17,
        "device": "cpu",
        "precision": "fp32",
        "deterministic": True,
        "log_every": 1,
        "eval_every": total_steps,
    }
    payload["checkpoint"] = {"save_every": total_steps, "keep_last": 1}
    return StabilityExperimentConfig.from_dict(payload)


def _trainer(root: Path) -> StabilityTrainer:
    config = _config()
    seed_everything(config.runtime.seed, deterministic=True)
    model = ControlledRecurrentTransformer(config.model)
    pattern = [4, 5, 6, 7, 8, 7, 6, 5]
    dataset = TokenBlockDataset(
        pattern * 32,
        sequence_length=config.data.sequence_length,
        stride=config.data.stride,
    )
    return StabilityTrainer(
        config=config,
        model=model,
        tokenizer_identity=ByteTokenizer().identity,
        train_dataset=dataset,
        eval_dataset=dataset,
        run=create_run(config, root / "runs"),
    )


def test_stability_optimizer_contains_every_parameter_once(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    optimizer_ids = [
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    ]
    model_ids = {
        id(parameter) for parameter in trainer.model.parameters() if parameter.requires_grad
    }

    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == model_ids


def test_stability_trainer_records_state_gate_gradient_and_compute_data(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path)

    result = trainer.train(max_steps=1)

    records = [
        json.loads(line)
        for line in trainer.run.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.steps_completed == 1
    assert len(result.last_iteration_diagnostics) == 2
    assert len(result.last_gate_diagnostics) == 2
    assert len(records) == 1
    record = records[0]
    assert record["variant"] == "S2"
    assert record["recurrent_iterations"] == 2
    assert len(record["recurrent_diagnostics"]) == 2
    assert len(record["gate_diagnostics"]) == 2
    assert {item["collapse"] for item in record["gate_diagnostics"]} <= {
        "zero",
        "one",
        "none",
    }
    assert record["gradient_norm"] > 0.0
    assert record["core_gradient_l2_norm"] > 0.0
    assert record["core_gradient_finite"] is True
    counts = count_parameters(trainer.model)
    assert record["total_parameters"] == counts.total
    assert record["trainable_parameters"] == counts.trainable
    assert record["active_parameters"] == counts.active
    assert record["active_flops_per_token"] > 0.0
    assert record["estimated_training_flops"] > 0
