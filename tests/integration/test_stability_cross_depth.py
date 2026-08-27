from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from aurora.cli.stability_cross_depth import evaluate_cross_depth
from aurora.data.dataset import TokenBlockDataset
from aurora.experiment import create_run
from aurora.model.recurrent_stability import ControlledRecurrentTransformer
from aurora.stability_config import StabilityExperimentConfig
from aurora.training.reproducibility import seed_everything
from aurora.training.stability_trainer import StabilityTrainer
from tests.fixtures.byte_tokenizer import ByteTokenizer
from tests.stability_helpers import stability_experiment_payload


def _cross_depth_config() -> StabilityExperimentConfig:
    payload = stability_experiment_payload(
        num_iterations=4,
        gated_update=True,
        max_seq_len=8,
        d_model=24,
        d_ff=48,
    )
    payload["experiment"] = {
        "run_id": "EXP-9604-stability-cross-depth-test",
        "description": "Stability cross-depth test",
        "dataset_version": "test-pattern-v1",
    }
    payload["data"] = {"sequence_length": 8, "batch_size": 2, "stride": 4}
    payload["scheduler"] = {"warmup_steps": 0, "total_steps": 1, "min_lr_ratio": 0.2}
    payload["runtime"] = {
        "seed": 17,
        "device": "cpu",
        "precision": "fp32",
        "deterministic": True,
        "log_every": 1,
        "eval_every": 1,
    }
    payload["checkpoint"] = {"save_every": 1, "keep_last": 1}
    return StabilityExperimentConfig.from_dict(payload)


def test_cross_depth_evaluates_exact_checkpoint_without_mutation(tmp_path: Path) -> None:
    config = _cross_depth_config()
    tokenizer = ByteTokenizer()
    seed_everything(config.runtime.seed, deterministic=True)
    model = ControlledRecurrentTransformer(config.model)
    pattern = [4, 5, 6, 7, 8, 7, 6, 5]
    dataset = TokenBlockDataset(
        pattern * 32,
        sequence_length=config.data.sequence_length,
        stride=config.data.stride,
    )
    run = create_run(config, tmp_path / "runs")
    trainer = StabilityTrainer(
        config=config,
        model=model,
        tokenizer_identity=tokenizer.identity,
        train_dataset=dataset,
        eval_dataset=dataset,
        run=run,
    )
    trainer.train(max_steps=1)
    checkpoint_path = run.checkpoints_dir / "fixture.pt"
    trainer.save(checkpoint_path)
    before = {name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()}
    output_path = run.root / "cross-depth.json"

    result = evaluate_cross_depth(
        config=config,
        tokenizer=tokenizer,
        eval_text="alpha beta gamma delta.\n" * 32,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        depths=(1, 2, 3, 4, 6, 8),
    )

    assert result["source_run_id"] == config.experiment.run_id
    assert result["source_config_hash"] == config.sha256()
    assert len(result["source_checkpoint_sha256"]) == 64
    assert result["configured_training_depth"] == 4
    assert result["evaluated_depths"] == [1, 2, 3, 4, 6, 8]
    assert [item["inference_depth"] for item in result["results"]] == [1, 2, 3, 4, 6, 8]
    for depth_result in result["results"]:
        depth = depth_result["inference_depth"]
        assert math.isfinite(depth_result["evaluation_loss"])
        assert len(depth_result["recurrent_diagnostics"]) == depth
        assert len(depth_result["gate_diagnostics"]) == depth
    after = trainer.model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert config.model.num_iterations == 4
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
