from __future__ import annotations

import torch
from conftest import model_config

from aurora.model.attention import GroupedQueryAttention
from aurora.model.mlp import SwiGLU
from aurora.model.norms import RMSNorm
from aurora.model.rope import RotaryEmbedding


def test_rms_norm_matches_hand_derived_reference() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    norm = RMSNorm(4, eps=1e-6)
    expected = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-6)

    actual = norm(values)

    torch.testing.assert_close(actual, expected)


def test_rope_preserves_vector_norm_and_leaves_position_zero_unchanged() -> None:
    torch.manual_seed(2)
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn(1, 2, 5, 8)
    rope = RotaryEmbedding(head_dim=8, max_seq_len=8, base=10_000.0)

    rotated_query, rotated_key = rope(query, key)

    torch.testing.assert_close(rotated_query[..., 0, :], query[..., 0, :])
    torch.testing.assert_close(rotated_key[..., 0, :], key[..., 0, :])
    torch.testing.assert_close(rotated_query.float().norm(dim=-1), query.float().norm(dim=-1))
    torch.testing.assert_close(rotated_key.float().norm(dim=-1), key.float().norm(dim=-1))


def test_grouped_query_attention_has_expected_shape_and_gradients() -> None:
    module = GroupedQueryAttention(model_config())
    inputs = torch.randn(2, 8, 32, requires_grad=True)

    output = module(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_swiglu_preserves_batch_sequence_and_model_dimensions() -> None:
    module = SwiGLU(d_model=32, d_ff=64, dropout=0.0)
    inputs = torch.randn(2, 7, 32)

    output = module(inputs)

    assert output.shape == inputs.shape
