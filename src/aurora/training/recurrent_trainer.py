from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch

from aurora.data.dataset import TokenBlockDataset
from aurora.experiment import RunArtifacts, append_jsonl
from aurora.model.recurrent import FixedLoopRecurrentTransformer
from aurora.model.recurrent_diagnostics import RecurrentStateDiagnostics
from aurora.recurrent_config import RecurrentExperimentConfig
from aurora.tokenization.base import TokenizerIdentity
from aurora.training.metrics import estimate_recurrent_transformer_flops
from aurora.training.recurrent_metrics import (
    CoreGradientStatistics,
    collect_core_gradient_statistics,
)
from aurora.training.trainer import Trainer, TrainingResult


@dataclass(frozen=True)
class RecurrentTrainingResult(TrainingResult):
    gradient_norms: list[float]
    core_gradient_statistics: list[CoreGradientStatistics]
    last_iteration_diagnostics: tuple[RecurrentStateDiagnostics, ...]


class RecurrentTrainer(Trainer):
    """Task 05 trainer that observes fixed-loop state and core gradients."""

    config: RecurrentExperimentConfig
    model: FixedLoopRecurrentTransformer

    def __init__(
        self,
        *,
        config: RecurrentExperimentConfig,
        model: FixedLoopRecurrentTransformer,
        tokenizer_identity: TokenizerIdentity,
        train_dataset: TokenBlockDataset,
        eval_dataset: TokenBlockDataset,
        run: RunArtifacts,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            tokenizer_identity=tokenizer_identity,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            run=run,
        )

    @staticmethod
    def _require_finite_core_gradients(statistics: CoreGradientStatistics) -> None:
        values = (statistics.l2_norm, statistics.mean_abs, statistics.max_abs)
        if not statistics.finite or not all(math.isfinite(value) for value in values):
            raise FloatingPointError("non-finite or missing recurrent core gradients")

    def train(self, max_steps: int | None = None) -> RecurrentTrainingResult:
        target_step = self.config.scheduler.total_steps if max_steps is None else max_steps
        if not self.step < target_step <= self.config.scheduler.total_steps:
            raise ValueError("max_steps must be greater than current step and at most total_steps")
        self.model.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        core_statistics: list[CoreGradientStatistics] = []
        last_diagnostics: tuple[RecurrentStateDiagnostics, ...] = ()
        wall_start = time.perf_counter()
        while self.step < target_step:
            input_ids, targets = self._batch(self.train_dataset, self._next_indices())
            step_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = self.model(
                    input_ids,
                    targets=targets,
                    collect_diagnostics=True,
                )
            if output.loss is None:
                raise RuntimeError("training model did not return a loss")
            loss_value = self._finite_scalar("loss", output.loss)
            if self.scaler is None:
                output.loss.backward()
            else:
                self.scaler.scale(output.loss).backward()
                self.scaler.unscale_(self.optimizer)
            core_gradient = collect_core_gradient_statistics(self.model.core)
            self._require_finite_core_gradients(core_gradient)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.optimizer.grad_clip,
                error_if_nonfinite=True,
            )
            gradient_norm_value = self._finite_scalar("gradient norm", gradient_norm)
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.scheduler.step()
            self.step += 1
            batch_tokens = int(targets.numel())
            self.tokens_seen += batch_tokens
            elapsed = max(time.perf_counter() - step_start, 1e-12)
            losses.append(loss_value)
            gradient_norms.append(gradient_norm_value)
            core_statistics.append(core_gradient)
            last_diagnostics = output.iteration_diagnostics
            if self.step % self.config.runtime.log_every == 0:
                peak_vram = (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                )
                append_jsonl(
                    self.run.metrics_path,
                    {
                        "core_gradient_finite": core_gradient.finite,
                        "core_gradient_l2_norm": core_gradient.l2_norm,
                        "core_gradient_max_abs": core_gradient.max_abs,
                        "core_gradient_mean_abs": core_gradient.mean_abs,
                        "core_gradient_nonzero_fraction": core_gradient.nonzero_fraction,
                        "estimated_training_flops": estimate_recurrent_transformer_flops(
                            self.config.model,
                            batch_size=input_ids.shape[0],
                            sequence_length=input_ids.shape[1],
                            training=True,
                        ),
                        "gradient_norm": gradient_norm_value,
                        "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                        "loss": loss_value,
                        "peak_vram_bytes": peak_vram,
                        "recurrent_diagnostics": [
                            asdict(diagnostic) for diagnostic in output.iteration_diagnostics
                        ],
                        "recurrent_iterations": self.config.model.num_iterations,
                        "step": self.step,
                        "tokens_per_second": batch_tokens / elapsed,
                        "tokens_seen": self.tokens_seen,
                    },
                )
        wall_time = time.perf_counter() - wall_start
        return RecurrentTrainingResult(
            initial_loss=losses[0],
            final_loss=losses[-1],
            losses=losses,
            steps_completed=self.step,
            tokens_seen=self.tokens_seen,
            wall_time_seconds=wall_time,
            gradient_norms=gradient_norms,
            core_gradient_statistics=core_statistics,
            last_iteration_diagnostics=last_diagnostics,
        )
