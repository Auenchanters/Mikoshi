from __future__ import annotations

from typing import Any

from aurora.recurrent_config import RecurrentExperimentConfig, RecurrentModelConfig
from tests.conftest import tiny_config_dict


def recurrent_model_config(*, num_iterations: int = 4, **overrides: Any) -> RecurrentModelConfig:
    values: dict[str, Any] = {
        "vocab_size": 64,
        "max_seq_len": 16,
        "d_model": 32,
        "num_heads": 4,
        "num_kv_heads": 2,
        "d_ff": 64,
        "num_prelude_layers": 1,
        "num_core_layers": 1,
        "num_coda_layers": 1,
        "num_iterations": num_iterations,
        "max_iterations": 8,
        "dropout": 0.0,
        "rope_base": 10_000.0,
        "qk_norm": True,
        "tie_embeddings": True,
        "norm_eps": 1e-6,
    }
    values.update(overrides)
    return RecurrentModelConfig(**values)


def recurrent_experiment_config(
    *, num_iterations: int = 4, total_steps: int = 8
) -> RecurrentExperimentConfig:
    payload = tiny_config_dict(
        experiment={
            "run_id": f"EXP-950{num_iterations}-recurrent-test",
            "description": f"Test recurrent R={num_iterations}",
            "dataset_version": "test-pattern-v1",
        },
        data={"sequence_length": 8, "batch_size": 2, "stride": 4},
        scheduler={"warmup_steps": 0, "total_steps": total_steps, "min_lr_ratio": 0.2},
        checkpoint={"save_every": total_steps, "keep_last": 1},
    )
    payload["model"] = {
        "vocab_size": 300,
        "max_seq_len": 8,
        "d_model": 24,
        "num_heads": 4,
        "num_kv_heads": 2,
        "d_ff": 48,
        "num_prelude_layers": 1,
        "num_core_layers": 1,
        "num_coda_layers": 1,
        "num_iterations": num_iterations,
        "max_iterations": 8,
        "dropout": 0.0,
        "rope_base": 10_000.0,
        "qk_norm": True,
        "tie_embeddings": True,
        "norm_eps": 1e-6,
    }
    return RecurrentExperimentConfig.from_dict(payload)
