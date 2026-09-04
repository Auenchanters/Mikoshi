from __future__ import annotations

from typing import Any

from aurora.stability_config import StabilityModelConfig


def stability_model_config(
    *,
    num_iterations: int = 4,
    input_anchoring: bool = False,
    gated_update: bool = False,
    state_stabilization: str = "none",
) -> StabilityModelConfig:
    return StabilityModelConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        num_prelude_layers=1,
        num_core_layers=1,
        num_coda_layers=1,
        num_iterations=num_iterations,
        max_iterations=8,
        dropout=0.0,
        rope_base=10_000.0,
        qk_norm=True,
        tie_embeddings=True,
        norm_eps=1e-6,
        input_anchoring=input_anchoring,
        gated_update=gated_update,
        state_stabilization=state_stabilization,
    )


def stability_experiment_payload(**model_overrides: Any) -> dict[str, object]:
    return {
        "experiment": {
            "run_id": "EXP-0600-stability-config",
            "description": "Stability configuration test",
            "dataset_version": "tiny-corpus-v1",
        },
        "tokenizer": {"kind": "bpe", "vocab_size": 300, "min_frequency": 2},
        "data": {"sequence_length": 16, "batch_size": 2, "stride": 16},
        "model": {
            "vocab_size": 300,
            "max_seq_len": 16,
            "d_model": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "d_ff": 64,
            "num_prelude_layers": 1,
            "num_core_layers": 1,
            "num_coda_layers": 1,
            "num_iterations": 4,
            "max_iterations": 8,
            "dropout": 0.0,
            "rope_base": 10_000.0,
            "qk_norm": True,
            "tie_embeddings": True,
            "norm_eps": 1e-6,
            "input_anchoring": False,
            "gated_update": False,
            "state_stabilization": "none",
            **model_overrides,
        },
        "optimizer": {
            "learning_rate": 0.001,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
        "scheduler": {"warmup_steps": 2, "total_steps": 20, "min_lr_ratio": 0.1},
        "runtime": {
            "seed": 17,
            "device": "cpu",
            "precision": "fp32",
            "deterministic": True,
            "log_every": 1,
            "eval_every": 10,
        },
        "checkpoint": {"save_every": 10, "keep_last": 2},
    }
