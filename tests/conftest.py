from __future__ import annotations

from typing import Any

from aurora.config import ModelConfig


def tiny_config_dict(**overrides: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "experiment": {
            "run_id": "EXP-0001-dense-sanity",
            "description": "Tiny dense baseline sanity run",
            "dataset_version": "tiny-corpus-v1",
        },
        "tokenizer": {
            "kind": "bpe",
            "vocab_size": 300,
            "min_frequency": 2,
        },
        "data": {
            "sequence_length": 16,
            "batch_size": 2,
            "stride": 16,
        },
        "model": {
            "vocab_size": 300,
            "max_seq_len": 16,
            "d_model": 32,
            "num_layers": 2,
            "num_heads": 4,
            "num_kv_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "rope_base": 10000.0,
            "qk_norm": True,
            "tie_embeddings": True,
        },
        "optimizer": {
            "learning_rate": 0.001,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
        "scheduler": {
            "warmup_steps": 2,
            "total_steps": 20,
            "min_lr_ratio": 0.1,
        },
        "runtime": {
            "seed": 17,
            "device": "cpu",
            "precision": "fp32",
            "deterministic": True,
            "log_every": 1,
            "eval_every": 10,
        },
        "checkpoint": {
            "save_every": 10,
            "keep_last": 2,
        },
    }
    for section, values in overrides.items():
        if section not in config:
            config[section] = values
        else:
            config[section].update(values)
    return config


def model_config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "vocab_size": 64,
        "max_seq_len": 16,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "d_ff": 64,
        "dropout": 0.0,
        "rope_base": 10_000.0,
        "qk_norm": True,
        "tie_embeddings": True,
        "norm_eps": 1e-6,
    }
    values.update(overrides)
    return ModelConfig(**values)
