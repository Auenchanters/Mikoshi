from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aurora.training.checkpoint import TrainingState, load_checkpoint, save_checkpoint


def training_state() -> TrainingState:
    return TrainingState(
        model={"weight": torch.tensor([[1.0, 2.0]])},
        optimizer={"state": {}, "param_groups": [{"lr": 0.001}]},
        scheduler={"last_epoch": 3},
        scaler=None,
        rng={"torch_cpu": torch.tensor([1, 2, 3], dtype=torch.uint8)},
        sampler={"epoch": 1, "cursor": 4},
        step=3,
        tokens_seen=96,
        config={"model": {"d_model": 32}},
        config_hash="a" * 64,
        tokenizer={"kind": "bpe", "sha256": "b" * 64},
        experiment={"run_id": "EXP-0001-dense-sanity"},
    )


def test_checkpoint_round_trip_restores_all_declared_keys(tmp_path: Path) -> None:
    path = tmp_path / "step.pt"

    save_checkpoint(path, training_state())
    payload = load_checkpoint(path)

    assert set(payload) == {
        "config",
        "config_hash",
        "experiment",
        "format_version",
        "model",
        "optimizer",
        "rng",
        "sampler",
        "scaler",
        "scheduler",
        "step",
        "tokenizer",
        "tokens_seen",
    }
    torch.testing.assert_close(payload["model"]["weight"], torch.tensor([[1.0, 2.0]]))
    assert payload["step"] == 3
    assert not (tmp_path / "step.pt.tmp").exists()


def test_checkpoint_loader_rejects_unknown_format_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    payload = training_state().to_payload()
    payload["format_version"] = 99
    torch.save(payload, path)

    with pytest.raises(ValueError, match="format version"):
        load_checkpoint(path)
