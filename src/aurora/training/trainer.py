from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from aurora.config import ExperimentConfig
from aurora.data.dataset import TokenBlockDataset
from aurora.data.sampler import DeterministicBatchSampler
from aurora.experiment import RunArtifacts, append_jsonl
from aurora.model.transformer import DenseTransformer
from aurora.tokenization.base import TokenizerIdentity
from aurora.training.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from aurora.training.metrics import count_parameters, estimate_transformer_flops
from aurora.training.reproducibility import capture_rng_state, restore_rng_state


@dataclass(frozen=True)
class TrainingResult:
    initial_loss: float
    final_loss: float
    losses: list[float]
    steps_completed: int
    tokens_seen: int
    wall_time_seconds: float


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    target_tokens: int
    batches: int


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


class Trainer:
    """Correctness-first single-device trainer for the dense control model."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        model: DenseTransformer,
        tokenizer_identity: TokenizerIdentity,
        train_dataset: TokenBlockDataset,
        eval_dataset: TokenBlockDataset,
        run: RunArtifacts,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer_identity = tokenizer_identity
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.run = run
        self.device = self._resolve_device(config.runtime.device)
        if self.device.type == "cpu" and config.runtime.precision != "fp32":
            raise ValueError("CPU training supports fp32 only")
        if (
            config.runtime.precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("CUDA device does not support bf16")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_multiplier,
        )
        use_scaler = self.device.type == "cuda" and config.runtime.precision == "fp16"
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None
        )
        self.sampler = DeterministicBatchSampler(
            len(train_dataset),
            batch_size=config.data.batch_size,
            seed=config.runtime.seed,
            drop_last=True,
        )
        self._train_iterator: Iterator[list[int]] = iter(self.sampler)
        self.step = 0
        self.tokens_seen = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._write_environment_metadata()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        return torch.device(requested)

    def _lr_multiplier(self, scheduler_step: int) -> float:
        scheduler_config = self.config.scheduler
        if scheduler_config.warmup_steps > 0 and scheduler_step < scheduler_config.warmup_steps:
            return (scheduler_step + 1) / scheduler_config.warmup_steps
        decay_steps = scheduler_config.total_steps - scheduler_config.warmup_steps
        if decay_steps <= 0:
            return scheduler_config.min_lr_ratio
        progress = min(
            max((scheduler_step - scheduler_config.warmup_steps) / decay_steps, 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return scheduler_config.min_lr_ratio + (1.0 - scheduler_config.min_lr_ratio) * cosine

    def _autocast(self) -> torch.autocast:
        enabled = self.device.type == "cuda" and self.config.runtime.precision != "fp32"
        dtype = torch.bfloat16 if self.config.runtime.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def _next_indices(self) -> list[int]:
        while True:
            try:
                return next(self._train_iterator)
            except StopIteration:
                self._train_iterator = iter(self.sampler)

    def _batch(self, dataset: TokenBlockDataset, indices: Sequence[int]) -> tuple[Tensor, Tensor]:
        examples = [dataset[index] for index in indices]
        inputs = torch.stack([example[0] for example in examples]).to(self.device)
        targets = torch.stack([example[1] for example in examples]).to(self.device)
        return inputs, targets

    @staticmethod
    def _finite_scalar(name: str, value: Tensor) -> float:
        if value.numel() != 1 or not bool(torch.isfinite(value).item()):
            raise FloatingPointError(f"non-finite {name}")
        return float(value.detach().item())

    def train(self, max_steps: int | None = None) -> TrainingResult:
        target_step = self.config.scheduler.total_steps if max_steps is None else max_steps
        if not self.step < target_step <= self.config.scheduler.total_steps:
            raise ValueError("max_steps must be greater than current step and at most total_steps")
        self.model.train()
        losses: list[float] = []
        wall_start = time.perf_counter()
        while self.step < target_step:
            input_ids, targets = self._batch(self.train_dataset, self._next_indices())
            step_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = self.model(input_ids, targets=targets)
            if output.loss is None:
                raise RuntimeError("training model did not return a loss")
            loss_value = self._finite_scalar("loss", output.loss)
            if self.scaler is None:
                output.loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.optimizer.grad_clip,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
            else:
                self.scaler.scale(output.loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.optimizer.grad_clip,
                    error_if_nonfinite=True,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            gradient_norm_value = self._finite_scalar("gradient norm", gradient_norm)
            self.scheduler.step()
            self.step += 1
            batch_tokens = int(targets.numel())
            self.tokens_seen += batch_tokens
            elapsed = max(time.perf_counter() - step_start, 1e-12)
            losses.append(loss_value)
            if self.step % self.config.runtime.log_every == 0:
                peak_vram = (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                )
                append_jsonl(
                    self.run.metrics_path,
                    {
                        "estimated_training_flops": estimate_transformer_flops(
                            self.config.model,
                            batch_size=input_ids.shape[0],
                            sequence_length=input_ids.shape[1],
                            training=True,
                        ),
                        "gradient_norm": gradient_norm_value,
                        "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                        "loss": loss_value,
                        "peak_vram_bytes": peak_vram,
                        "step": self.step,
                        "tokens_per_second": batch_tokens / elapsed,
                        "tokens_seen": self.tokens_seen,
                    },
                )
        wall_time = time.perf_counter() - wall_start
        return TrainingResult(
            initial_loss=losses[0],
            final_loss=losses[-1],
            losses=losses,
            steps_completed=self.step,
            tokens_seen=self.tokens_seen,
            wall_time_seconds=wall_time,
        )

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> EvaluationResult:
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive")
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = 0
        try:
            for start in range(0, len(self.eval_dataset), self.config.data.batch_size):
                if max_batches is not None and batches >= max_batches:
                    break
                indices = list(
                    range(start, min(start + self.config.data.batch_size, len(self.eval_dataset)))
                )
                input_ids, targets = self._batch(self.eval_dataset, indices)
                with self._autocast():
                    output = self.model(input_ids, targets=targets)
                if output.loss is None:
                    raise RuntimeError("evaluation model did not return a loss")
                mean_loss = self._finite_scalar("evaluation loss", output.loss)
                token_count = int(targets.numel())
                total_loss += mean_loss * token_count
                total_tokens += token_count
                batches += 1
        finally:
            self.model.train(was_training)
        if total_tokens == 0:
            raise ValueError("evaluation dataset produced no target tokens")
        return EvaluationResult(
            loss=total_loss / total_tokens,
            target_tokens=total_tokens,
            batches=batches,
        )

    def save(self, path: Path) -> None:
        scaler_state = None if self.scaler is None else self.scaler.state_dict()
        state = TrainingState(
            model=cast(Mapping[str, object], self.model.state_dict()),
            optimizer=cast(Mapping[str, object], self.optimizer.state_dict()),
            scheduler=cast(Mapping[str, object], self.scheduler.state_dict()),
            scaler=cast(Mapping[str, object] | None, scaler_state),
            rng=capture_rng_state(),
            sampler=self.sampler.state_dict(),
            step=self.step,
            tokens_seen=self.tokens_seen,
            config=self.config.to_dict(),
            config_hash=self.config.sha256(),
            tokenizer=self.tokenizer_identity.to_dict(),
            experiment=asdict(self.config.experiment),
        )
        save_checkpoint(path, state)

    def resume(self, path: Path) -> None:
        payload = load_checkpoint(path, map_location=self.device)
        if payload["config_hash"] != self.config.sha256():
            raise ValueError("checkpoint configuration does not match trainer configuration")
        tokenizer = cast(Mapping[str, object], payload["tokenizer"])
        if tokenizer.get("sha256") != self.tokenizer_identity.sha256:
            raise ValueError("checkpoint tokenizer does not match trainer tokenizer")
        self.model.load_state_dict(cast(Mapping[str, Tensor], payload["model"]))
        self.optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer"]))
        self.scheduler.load_state_dict(cast(dict[str, Any], payload["scheduler"]))
        scaler_payload = payload["scaler"]
        if self.scaler is None:
            if scaler_payload is not None:
                raise ValueError("checkpoint has fp16 scaler state but trainer does not")
        else:
            if not isinstance(scaler_payload, Mapping):
                raise ValueError("fp16 trainer requires checkpoint scaler state")
            self.scaler.load_state_dict(cast(dict[str, Any], scaler_payload))
        self.sampler.load_state_dict(cast(Mapping[str, object], payload["sampler"]))
        self.step = cast(int, payload["step"])
        self.tokens_seen = cast(int, payload["tokens_seen"])
        restore_rng_state(cast(Mapping[str, object], payload["rng"]))
        self._train_iterator = iter(self.sampler)

    def _write_environment_metadata(self) -> None:
        counts = count_parameters(self.model)
        metadata = {
            "active_parameters": counts.active,
            "config_hash": self.config.sha256(),
            "dataset_version": self.config.experiment.dataset_version,
            "device": str(self.device),
            "git_commit": _git_commit(),
            "hardware": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else platform.processor() or platform.machine()
            ),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "precision": self.config.runtime.precision,
            "python_version": sys.version.split()[0],
            "run_id": self.config.experiment.run_id,
            "seed": self.config.runtime.seed,
            "status": "initialized",
            "tokenizer": self.tokenizer_identity.to_dict(),
            "torch_version": torch.__version__,
            "total_parameters": counts.total,
            "trainable_parameters": counts.trainable,
        }
        self.run.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
