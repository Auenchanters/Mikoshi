"""Reproducible single-device training infrastructure."""

from aurora.training.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from aurora.training.metrics import (
    ParameterCounts,
    count_parameters,
    estimate_recurrent_transformer_flops,
    estimate_transformer_flops,
)
from aurora.training.recurrent_trainer import RecurrentTrainer, RecurrentTrainingResult
from aurora.training.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from aurora.training.trainer import EvaluationResult, Trainer, TrainingResult

__all__ = [
    "EvaluationResult",
    "ParameterCounts",
    "RecurrentTrainer",
    "RecurrentTrainingResult",
    "Trainer",
    "TrainingResult",
    "TrainingState",
    "capture_rng_state",
    "count_parameters",
    "estimate_recurrent_transformer_flops",
    "estimate_transformer_flops",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "seed_everything",
]
