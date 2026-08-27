from __future__ import annotations

import json
from pathlib import Path

from aurora.cli.recurrent_tiny import (
    _dataset_hash as recurrent_dataset_hash,
)
from aurora.cli.recurrent_tiny import run_recurrent_pipeline
from aurora.cli.tiny import _dataset_hash as dense_dataset_hash
from aurora.experiment import create_run
from tests.fixtures.byte_tokenizer import ByteTokenizer
from tests.recurrent_helpers import recurrent_experiment_config


def test_dense_and_recurrent_runs_use_the_same_dataset_identity() -> None:
    train_text = "identical training corpus\n"
    eval_text = "identical evaluation corpus\n"

    assert recurrent_dataset_hash(train_text, eval_text) == dense_dataset_hash(
        train_text, eval_text
    )


def test_recurrent_pipeline_trains_evaluates_reloads_and_records_diagnostics(
    tmp_path: Path,
) -> None:
    config = recurrent_experiment_config(num_iterations=2, total_steps=8)
    run = create_run(config, tmp_path / "runs")
    text = "alpha beta gamma delta. alpha beta gamma delta.\n" * 64

    result = run_recurrent_pipeline(
        config=config,
        tokenizer=ByteTokenizer(),
        train_text=text,
        eval_text=text,
        run=run,
        generation_prompt="alpha beta",
        generation_tokens=4,
    )

    assert result["initial_loss"] > result["final_loss"]
    assert result["checkpoint_reloaded"] is True
    assert result["deterministic_replay_verified"] is True
    assert result["generated_text"]
    assert result["num_iterations"] == 2
    assert result["training_tokens"] == 8 * 2 * 8
    assert result["active_flops_per_token"] > 0.0
    assert result["estimated_training_flops"] > 0
    assert result["final_gradient_norm"] > 0.0
    assert result["final_core_gradient_statistics"]["l2_norm"] > 0.0
    assert len(result["final_recurrent_diagnostics"]) == 2
    assert result["total_parameters"] == result["active_parameters"]
    recorded = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert recorded == result
    assert (run.checkpoints_dir / "final.pt").exists()
