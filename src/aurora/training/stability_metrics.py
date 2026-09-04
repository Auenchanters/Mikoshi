from __future__ import annotations

from aurora.stability_config import StabilityModelConfig
from aurora.training.metrics import estimate_recurrent_transformer_flops


def estimate_stability_transformer_flops(
    config: StabilityModelConfig,
    batch_size: int,
    sequence_length: int,
    training: bool,
) -> int:
    """Estimate active Task 06 matmul FLOPs using the Task 05 convention."""

    forward = estimate_recurrent_transformer_flops(
        config.as_recurrent_config(),
        batch_size=batch_size,
        sequence_length=sequence_length,
        training=False,
    )
    tokens = batch_size * sequence_length
    per_iteration = 0
    if config.input_anchoring:
        per_iteration += 2 * tokens * config.d_model**2
    if config.gated_update:
        per_iteration += 4 * tokens * config.d_model
    forward += config.num_iterations * per_iteration
    return forward * 3 if training else forward
