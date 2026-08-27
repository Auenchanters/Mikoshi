from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml


class ConfigError(ValueError):
    """Raised when a configuration cannot be parsed or validated."""


@dataclass(frozen=True)
class ExperimentIdentityConfig:
    run_id: str
    description: str
    dataset_version: str


@dataclass(frozen=True)
class TokenizerConfig:
    kind: str
    vocab_size: int
    min_frequency: int
    normalization: str = "NFKC"
    pre_tokenizer: str = "byte_level"
    add_prefix_space: bool = False


@dataclass(frozen=True)
class DataConfig:
    sequence_length: int
    batch_size: int
    stride: int


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    d_ff: int
    dropout: float = 0.0
    rope_base: float = 10_000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    norm_eps: float = 1e-6


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    grad_clip: float


@dataclass(frozen=True)
class SchedulerConfig:
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    device: str
    precision: str
    deterministic: bool
    log_every: int
    eval_every: int


@dataclass(frozen=True)
class CheckpointConfig:
    save_every: int
    keep_last: int


ConfigSection = TypeVar("ConfigSection")


def _section(section_type: type[ConfigSection], value: object, path: str) -> ConfigSection:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    field_map = {field.name: field for field in fields(cast(Any, section_type))}
    unknown = sorted(set(value) - set(field_map))
    if unknown:
        raise ConfigError(f"{path} has unknown keys: {', '.join(str(key) for key in unknown)}")
    missing = sorted(
        name
        for name, field in field_map.items()
        if name not in value and field.default is MISSING and field.default_factory is MISSING
    )
    if missing:
        raise ConfigError(f"{path} is missing required keys: {', '.join(missing)}")
    kwargs = dict(value)
    if section_type is OptimizerConfig and "betas" in kwargs:
        betas = kwargs["betas"]
        if not isinstance(betas, list | tuple) or len(betas) != 2:
            raise ConfigError("optimizer.betas must contain exactly two numbers")
        kwargs["betas"] = (float(betas[0]), float(betas[1]))
    try:
        return section_type(**kwargs)
    except TypeError as error:
        raise ConfigError(f"invalid {path} configuration: {error}") from error


def _primitive(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_primitive(item) for item in value]
    return value


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or cast(int, value) <= 0:
        raise ConfigError(f"{name} must be a positive integer")


def _finite_positive(name: str, value: object, *, allow_zero: bool = False) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{name} must be a finite number")
    numeric = float(value)
    valid = numeric >= 0.0 if allow_zero else numeric > 0.0
    if not math.isfinite(numeric) or not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be a finite {qualifier} number")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentIdentityConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    runtime: RuntimeConfig
    checkpoint: CheckpointConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExperimentConfig:
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
            model=_section(ModelConfig, payload["model"], "model"),
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
    def from_yaml(cls, path: Path) -> ExperimentConfig:
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
        if not re.fullmatch(r"EXP-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*", self.experiment.run_id):
            raise ConfigError("experiment.run_id must match EXP-NNNN-slug")
        if not self.experiment.description.strip():
            raise ConfigError("experiment.description must not be empty")
        if not self.experiment.dataset_version.strip():
            raise ConfigError("experiment.dataset_version must not be empty")

        if self.tokenizer.kind != "bpe":
            raise ConfigError("tokenizer.kind must be 'bpe' in production configurations")
        _positive_int("tokenizer.vocab_size", self.tokenizer.vocab_size)
        if self.tokenizer.vocab_size < 260:
            raise ConfigError("tokenizer.vocab_size must be at least 260 for byte coverage")
        _positive_int("tokenizer.min_frequency", self.tokenizer.min_frequency)
        if self.tokenizer.normalization != "NFKC":
            raise ConfigError("tokenizer.normalization must be NFKC in this phase")
        if self.tokenizer.pre_tokenizer != "byte_level":
            raise ConfigError("tokenizer.pre_tokenizer must be byte_level in this phase")
        if self.tokenizer.add_prefix_space:
            raise ConfigError("tokenizer.add_prefix_space must be false in this phase")

        for name, value in (
            ("data.sequence_length", self.data.sequence_length),
            ("data.batch_size", self.data.batch_size),
            ("data.stride", self.data.stride),
            ("model.vocab_size", self.model.vocab_size),
            ("model.max_seq_len", self.model.max_seq_len),
            ("model.d_model", self.model.d_model),
            ("model.num_layers", self.model.num_layers),
            ("model.num_heads", self.model.num_heads),
            ("model.num_kv_heads", self.model.num_kv_heads),
            ("model.d_ff", self.model.d_ff),
            ("runtime.log_every", self.runtime.log_every),
            ("runtime.eval_every", self.runtime.eval_every),
            ("scheduler.total_steps", self.scheduler.total_steps),
            ("checkpoint.save_every", self.checkpoint.save_every),
            ("checkpoint.keep_last", self.checkpoint.keep_last),
        ):
            _positive_int(name, value)
        if type(self.scheduler.warmup_steps) is not int or self.scheduler.warmup_steps < 0:
            raise ConfigError("scheduler.warmup_steps must be a nonnegative integer")
        if self.scheduler.warmup_steps > self.scheduler.total_steps:
            raise ConfigError("scheduler.warmup_steps cannot exceed total_steps")
        if self.data.sequence_length > self.model.max_seq_len:
            raise ConfigError("data.sequence_length cannot exceed model.max_seq_len")
        if self.model.vocab_size != self.tokenizer.vocab_size:
            raise ConfigError("model.vocab_size must equal tokenizer.vocab_size")
        if self.model.d_model % self.model.num_heads != 0:
            raise ConfigError("model.d_model must be divisible by model.num_heads")
        if self.model.num_heads % self.model.num_kv_heads != 0:
            raise ConfigError("model.num_heads must be divisible by model.num_kv_heads")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")
        _finite_positive("model.rope_base", self.model.rope_base)
        _finite_positive("model.norm_eps", self.model.norm_eps)
        _finite_positive("optimizer.learning_rate", self.optimizer.learning_rate)
        _finite_positive("optimizer.weight_decay", self.optimizer.weight_decay, allow_zero=True)
        _finite_positive("optimizer.grad_clip", self.optimizer.grad_clip)
        if any(beta <= 0.0 or beta >= 1.0 for beta in self.optimizer.betas):
            raise ConfigError("optimizer.betas values must be in (0, 1)")
        if not 0.0 <= self.scheduler.min_lr_ratio <= 1.0:
            raise ConfigError("scheduler.min_lr_ratio must be in [0, 1]")
        if type(self.runtime.seed) is not int or self.runtime.seed < 0:
            raise ConfigError("runtime.seed must be a nonnegative integer")
        if self.runtime.device not in {"cpu", "cuda", "auto"}:
            raise ConfigError("runtime.device must be cpu, cuda, or auto")
        if self.runtime.precision not in {"fp32", "bf16", "fp16"}:
            raise ConfigError("runtime.precision must be fp32, bf16, or fp16")
        if self.runtime.device == "cpu" and self.runtime.precision != "fp32":
            raise ConfigError("CPU runtime requires fp32 precision")
