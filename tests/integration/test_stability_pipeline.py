from __future__ import annotations

import json
from pathlib import Path

from aurora.cli.stability_tiny import run_stability_pipeline
from aurora.experiment import create_run
from aurora.stability_config import StabilityExperimentConfig
from tests.fixtures.byte_tokenizer import ByteTokenizer
from tests.stability_helpers import stability_experiment_payload


def _pipeline_config() -> StabilityExperimentConfig:
    payload = stability_experiment_payload(
        num_iterations=2,
        gated_update=True,
        max_seq_len=8,
        d_model=24,
        d_ff=48,
    )
    payload["experiment"] = {
        "run_id": "EXP-9603-stability-pipeline-test",
        "description": "Stability pipeline test",
        "dataset_version": "test-pattern-v1",
    }
    payload["data"] = {"sequence_length": 8, "batch_size": 2, "stride": 4}
    payload["optimizer"] = {
        "learning_rate": 0.01,
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "grad_clip": 1.0,
    }
    payload["scheduler"] = {"warmup_steps": 0, "total_steps": 8, "min_lr_ratio": 0.2}
    payload["runtime"] = {
        "seed": 17,
        "device": "cpu",
        "precision": "fp32",
        "deterministic": True,
        "log_every": 1,
        "eval_every": 8,
    }
    payload["checkpoint"] = {"save_every": 8, "keep_last": 1}
    return StabilityExperimentConfig.from_dict(payload)


def test_stability_pipeline_trains_reloads_generates_and_records(tmp_path: Path) -> None:
    config = _pipeline_config()
    run = create_run(config, tmp_path / "runs")
    text = "alpha beta gamma delta. alpha beta gamma delta.\n" * 64

    result = run_stability_pipeline(
        config=config,
        tokenizer=ByteTokenizer(),
        train_text=text,
        eval_text=text,
        run=run,
        generation_prompt="alpha beta",
        generation_tokens=4,
    )

    assert result["variant"] == "S2"
    assert result["input_anchoring"] is False
    assert result["gated_update"] is True
    assert result["state_stabilization"] == "none"
    assert result["initial_loss"] > result["final_loss"]
    assert result["evaluation_loss"] > 0.0
    assert result["checkpoint_reloaded"] is True
    assert result["deterministic_replay_verified"] is True
    assert result["generated_text"]
    assert result["training_tokens"] == 8 * 2 * 8
    assert result["active_flops_per_token"] > 0.0
    assert result["estimated_training_flops"] > 0
    assert result["total_parameters"] == result["active_parameters"]
    assert len(result["final_recurrent_diagnostics"]) == 2
    assert len(result["final_gate_diagnostics"]) == 2
    assert result["gate_collapse_status"] in {"zero", "one", "none"}
    assert len(result["config_hash"]) == 64
    assert len(result["dataset_sha256"]) == 64
    assert len(result["tokenizer_sha256"]) == 64
    assert json.loads(run.result_path.read_text(encoding="utf-8")) == result
    assert (run.checkpoints_dir / "final.pt").exists()
