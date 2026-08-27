from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from aurora.config import (
    CheckpointConfig,
    ConfigError,
    DataConfig,
    ExperimentIdentityConfig,
    OptimizerConfig,
    RuntimeConfig,
    SchedulerConfig,
    TokenizerConfig,
    _primitive,
    _section,
)
from aurora.recurrent_config import RecurrentExperimentConfig, RecurrentModelConfig

_STABILITY_VARIANTS = {
    (False, False, "none"): "S0",
    (True, False, "none"): "S1",
    (False, True, "none"): "S2",
    (False, False, "initial_rms"): "S3",
    (True, True, "none"): "S4",
}


@dataclass(frozen=True)
class StabilityModelConfig(RecurrentModelConfig):
    input_anchoring: bool = False
    gated_update: bool = False
    state_stabilization: str = "none"

    def as_recurrent_config(self) -> RecurrentModelConfig:
        return RecurrentModelConfig(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            d_ff=self.d_ff,
            num_prelude_layers=self.num_prelude_layers,
            num_core_layers=self.num_core_layers,
            num_coda_layers=self.num_coda_layers,
            num_iterations=self.num_iterations,
            max_iterations=self.max_iterations,
            dropout=self.dropout,
            rope_base=self.rope_base,
            qk_norm=self.qk_norm,
            tie_embeddings=self.tie_embeddings,
            norm_eps=self.norm_eps,
        )

    @property
    def variant(self) -> str:
        try:
            return _STABILITY_VARIANTS[
                (self.input_anchoring, self.gated_update, self.state_stabilization)
            ]
        except KeyError as error:
            raise ConfigError("unsupported stability variant combination") from error


@dataclass(frozen=True)
class StabilityExperimentConfig(RecurrentExperimentConfig):
    model: StabilityModelConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> StabilityExperimentConfig:
        allowed = {
            "experiment",
            "tokenizer",
            "data",
            "model",
            "optimizer",
            "scheduler",
            "runtime",
            "checkpoint",
            "config_hash",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ConfigError(f"unknown top-level keys: {', '.join(unknown)}")
        required = allowed - {"config_hash"}
        missing = sorted(required - set(payload))
        if missing:
            raise ConfigError(f"missing top-level keys: {', '.join(missing)}")
        config = cls(
            experiment=_section(ExperimentIdentityConfig, payload["experiment"], "experiment"),
            tokenizer=_section(TokenizerConfig, payload["tokenizer"], "tokenizer"),
            data=_section(DataConfig, payload["data"], "data"),
            model=_section(StabilityModelConfig, payload["model"], "model"),
            optimizer=_section(OptimizerConfig, payload["optimizer"], "optimizer"),
            scheduler=_section(SchedulerConfig, payload["scheduler"], "scheduler"),
            runtime=_section(RuntimeConfig, payload["runtime"], "runtime"),
            checkpoint=_section(CheckpointConfig, payload["checkpoint"], "checkpoint"),
        )
        config.validate()
        expected_hash = payload.get("config_hash")
        if expected_hash is not None and expected_hash != config.sha256():
            raise ConfigError("config_hash does not match the resolved configuration")
        return config

    @classmethod
    def from_yaml(cls, path: Path) -> StabilityExperimentConfig:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ConfigError(f"could not read configuration {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise ConfigError("configuration root must be a mapping")
        return cls.from_dict(cast(Mapping[str, object], payload))

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _primitive(self))

    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        RecurrentExperimentConfig(
            experiment=self.experiment,
            tokenizer=self.tokenizer,
            data=self.data,
            model=self.model.as_recurrent_config(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            runtime=self.runtime,
            checkpoint=self.checkpoint,
        ).validate()
        if type(self.model.input_anchoring) is not bool:
            raise ConfigError("model.input_anchoring must be a boolean")
        if type(self.model.gated_update) is not bool:
            raise ConfigError("model.gated_update must be a boolean")
        if type(self.model.state_stabilization) is not str:
            raise ConfigError("model.state_stabilization must be a string")
        settings = (
            self.model.input_anchoring,
            self.model.gated_update,
            self.model.state_stabilization,
        )
        if settings not in _STABILITY_VARIANTS:
            raise ConfigError("unsupported stability variant combination")
