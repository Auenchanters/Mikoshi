from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aurora.config import ConfigError
from aurora.recurrent_config import RecurrentExperimentConfig
from tests.conftest import tiny_config_dict


def recurrent_config_dict(*, num_iterations: int = 4) -> dict[str, Any]:
    payload = tiny_config_dict()
    payload["experiment"] = {
        "run_id": f"EXP-050{num_iterations}-recurrent-{num_iterations}",
        "description": f"Fixed-loop recurrent R={num_iterations}",
        "dataset_version": "tiny-corpus-v1",
    }
    payload["model"] = {
        "vocab_size": 300,
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
    return payload


@pytest.mark.parametrize("num_iterations", [1, 2, 4, 8])
def test_recurrent_config_accepts_supported_fixed_iterations(num_iterations: int) -> None:
    config = RecurrentExperimentConfig.from_dict(
        recurrent_config_dict(num_iterations=num_iterations)
    )

    assert config.model.num_iterations == num_iterations
    assert config.model.max_iterations == 8


def test_recurrent_config_yaml_round_trip_and_hash(tmp_path: Path) -> None:
    config = RecurrentExperimentConfig.from_dict(recurrent_config_dict())
    path = tmp_path / "recurrent.yaml"

    config.to_yaml(path, include_hash=True)
    loaded = RecurrentExperimentConfig.from_yaml(path)

    assert loaded == config
    assert loaded.sha256() == config.sha256()


@pytest.mark.parametrize("num_iterations", [0, 3, 9])
def test_recurrent_config_rejects_unsupported_iterations(num_iterations: int) -> None:
    with pytest.raises(ConfigError, match="num_iterations"):
        RecurrentExperimentConfig.from_dict(recurrent_config_dict(num_iterations=num_iterations))


def test_recurrent_config_rejects_iteration_count_above_maximum() -> None:
    payload = recurrent_config_dict(num_iterations=8)
    payload["model"]["max_iterations"] = 4

    with pytest.raises(ConfigError, match="max_iterations"):
        RecurrentExperimentConfig.from_dict(payload)


def test_recurrent_config_rejects_empty_stage() -> None:
    payload = recurrent_config_dict()
    payload["model"]["num_core_layers"] = 0

    with pytest.raises(ConfigError, match="num_core_layers"):
        RecurrentExperimentConfig.from_dict(payload)


def test_recurrent_config_rejects_unknown_model_key() -> None:
    payload = recurrent_config_dict()
    payload["model"]["adaptive_halting"] = True

    with pytest.raises(ConfigError, match="unknown"):
        RecurrentExperimentConfig.from_dict(payload)
