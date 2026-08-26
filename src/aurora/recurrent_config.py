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
    ExperimentConfig,
    ExperimentIdentityConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SchedulerConfig,
    TokenizerConfig,
    _positive_int,
    _primitive,
    _section,
)

SUPPORTED_ITERATIONS = frozenset({1, 2, 4, 8})


@dataclass(frozen=True)
class RecurrentModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    num_kv_heads: int
    d_ff: int
    num_prelude_layers: int
    num_core_layers: int
    num_coda_layers: int
    num_iterations: int
    max_iterations: int
    dropout: float = 0.0
    rope_base: float = 10_000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    norm_eps: float = 1e-6

    def as_block_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            num_layers=(self.num_prelude_layers + self.num_core_layers + self.num_coda_layers),
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            rope_base=self.rope_base,
            qk_norm=self.qk_norm,
            tie_embeddings=self.tie_embeddings,
            norm_eps=self.norm_eps,
        )


@dataclass(frozen=True)
class RecurrentExperimentConfig:
    experiment: ExperimentIdentityConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    model: RecurrentModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    runtime: RuntimeConfig
    checkpoint: CheckpointConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RecurrentExperimentConfig:
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
            model=_section(RecurrentModelConfig, payload["model"], "model"),
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
    def from_yaml(cls, path: Path) -> RecurrentExperimentConfig:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ConfigError(f"could not read configuration {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise ConfigError("configuration root must be a mapping")
        return cls.from_dict(cast(Mapping[str, object], payload))

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _primitive(self))

    def to_yaml(self, path: Path, *, include_hash: bool = False) -> None:
        payload = self.to_dict()
        if include_hash:
            payload = {"config_hash": self.sha256(), **payload}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )

    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        dense_shape = ExperimentConfig(
            experiment=self.experiment,
            tokenizer=self.tokenizer,
            data=self.data,
            model=self.model.as_block_config(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            runtime=self.runtime,
            checkpoint=self.checkpoint,
        )
        dense_shape.validate()
        for name, value in (
            ("model.num_prelude_layers", self.model.num_prelude_layers),
            ("model.num_core_layers", self.model.num_core_layers),
            ("model.num_coda_layers", self.model.num_coda_layers),
            ("model.max_iterations", self.model.max_iterations),
        ):
            _positive_int(name, value)
        if type(self.model.num_iterations) is not int or (
            self.model.num_iterations not in SUPPORTED_ITERATIONS
        ):
            raise ConfigError("model.num_iterations must be one of 1, 2, 4, or 8")
        if self.model.max_iterations != 8:
            raise ConfigError("model.max_iterations must be 8 for Task 05")
        if self.model.num_iterations > self.model.max_iterations:
            raise ConfigError("model.num_iterations cannot exceed model.max_iterations")
