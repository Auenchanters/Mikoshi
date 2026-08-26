from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_KEYS = {
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


@dataclass(frozen=True)
class TrainingState:
    model: Mapping[str, object]
    optimizer: Mapping[str, object]
    scheduler: Mapping[str, object]
    scaler: Mapping[str, object] | None
    rng: Mapping[str, object]
    sampler: Mapping[str, object]
    step: int
    tokens_seen: int
    config: Mapping[str, object]
    config_hash: str
    tokenizer: Mapping[str, object]
    experiment: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "config": dict(self.config),
            "config_hash": self.config_hash,
            "experiment": dict(self.experiment),
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model": dict(self.model),
            "optimizer": dict(self.optimizer),
            "rng": dict(self.rng),
            "sampler": dict(self.sampler),
            "scaler": None if self.scaler is None else dict(self.scaler),
            "scheduler": dict(self.scheduler),
            "step": self.step,
            "tokenizer": dict(self.tokenizer),
            "tokens_seen": self.tokens_seen,
        }


def save_checkpoint(path: Path, state: TrainingState) -> None:
    if state.step < 0 or state.tokens_seen < 0:
        raise ValueError("checkpoint counters must be nonnegative")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state.to_payload(), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    raw = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint payload must be a mapping")
    payload = cast(dict[str, object], raw)
    if set(payload) != CHECKPOINT_KEYS:
        raise ValueError("checkpoint has missing or unknown keys")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format version: {payload['format_version']}")
    for counter in ("step", "tokens_seen"):
        value = payload[counter]
        if type(value) is not int or cast(int, value) < 0:
            raise ValueError(f"checkpoint {counter} must be a nonnegative integer")
    config_hash = payload["config_hash"]
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError("checkpoint config_hash must be a SHA-256 string")
    tokenizer = payload["tokenizer"]
    if not isinstance(tokenizer, Mapping):
        raise ValueError("checkpoint tokenizer identity must be a mapping")
    tokenizer_hash = tokenizer.get("sha256")
    if not isinstance(tokenizer_hash, str) or len(tokenizer_hash) != 64:
        raise ValueError("checkpoint tokenizer identity must contain a SHA-256 hash")
    return payload
