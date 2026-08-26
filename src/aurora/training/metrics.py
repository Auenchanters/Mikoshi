from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from aurora.config import ModelConfig


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int
    active: int


def count_parameters(model: nn.Module) -> ParameterCounts:
    seen: set[int] = set()
    total = 0
    trainable = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return ParameterCounts(total=total, trainable=trainable, active=trainable)


def estimate_transformer_flops(
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
    training: bool,
) -> int:
    """Estimate matmul FLOPs; normalization, activation, and softmax are omitted."""

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if sequence_length > config.max_seq_len:
        raise ValueError("sequence_length exceeds model.max_seq_len")
    head_dim = config.d_model // config.num_heads
    kv_dim = config.num_kv_heads * head_dim
    tokens = batch_size * sequence_length
    qkv_projection = 2 * tokens * (config.d_model * config.d_model + 2 * config.d_model * kv_dim)
    output_projection = 2 * tokens * config.d_model * config.d_model
    attention_products = 4 * batch_size * config.num_heads * sequence_length**2 * head_dim
    swiglu_projections = 6 * tokens * config.d_model * config.d_ff
    per_layer = qkv_projection + output_projection + attention_products + swiglu_projections
    vocabulary_projection = 2 * tokens * config.d_model * config.vocab_size
    forward = config.num_layers * per_layer + vocabulary_projection
    return forward * 3 if training else forward
