from __future__ import annotations

import json
from pathlib import Path

from conftest import tiny_config_dict

from aurora.cli.tiny import run_tiny_pipeline
from aurora.config import ExperimentConfig
from aurora.experiment import create_run
from tests.fixtures.byte_tokenizer import ByteTokenizer


def test_tiny_pipeline_trains_evaluates_checkpoints_reloads_and_generates(
    tmp_path: Path,
) -> None:
    payload = tiny_config_dict(
        experiment={"run_id": "EXP-9000-integration"},
        tokenizer={"vocab_size": 260},
        data={"sequence_length": 8, "batch_size": 4, "stride": 4},
        model={
            "vocab_size": 260,
            "max_seq_len": 8,
            "d_model": 24,
            "num_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 2,
            "d_ff": 48,
        },
        optimizer={"learning_rate": 0.01, "weight_decay": 0.0},
        scheduler={"warmup_steps": 0, "total_steps": 8, "min_lr_ratio": 0.2},
        checkpoint={"save_every": 8, "keep_last": 1},
    )
    config = ExperimentConfig.from_dict(payload)
    run = create_run(config, tmp_path / "runs")
    text = "alpha beta gamma delta. alpha beta gamma delta.\n" * 64

    result = run_tiny_pipeline(
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
    assert result["training_tokens"] == 8 * 4 * 8
    assert result["total_parameters"] == result["active_parameters"]
    assert result["dataset_sha256"]
    recorded = json.loads(run.result_path.read_text(encoding="utf-8"))
    assert recorded == result
    assert (run.checkpoints_dir / "final.pt").exists()
