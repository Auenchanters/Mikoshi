from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional
from conftest import model_config

from aurora.model.transformer import DenseTransformer


def test_dense_transformer_returns_aligned_causal_lm_loss() -> None:
    torch.manual_seed(3)
    model = DenseTransformer(model_config())
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    target_ids = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])

    output = model(input_ids, targets=target_ids)

    assert output.loss is not None
    expected = functional.cross_entropy(
        output.logits.reshape(-1, 64),
        target_ids.reshape(-1),
    )
    torch.testing.assert_close(output.loss, expected)
    assert output.logits.shape == (2, 4, 64)


def test_dense_transformer_ties_embedding_and_lm_head_weights() -> None:
    model = DenseTransformer(model_config(tie_embeddings=True))

    assert model.lm_head.weight is model.token_embedding.weight


def test_dense_transformer_rejects_sequence_beyond_configured_limit() -> None:
    model = DenseTransformer(model_config(max_seq_len=4))

    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.ones((1, 5), dtype=torch.long))


def test_greedy_generation_is_deterministic_and_extends_prompt() -> None:
    torch.manual_seed(4)
    model = DenseTransformer(model_config()).eval()
    prompt = torch.tensor([[1, 5, 6]])

    first = model.generate(prompt, max_new_tokens=4)
    second = model.generate(prompt, max_new_tokens=4)

    assert torch.equal(first, second)
    assert first.shape == (1, 7)
    assert torch.equal(first[:, :3], prompt)


def test_seeded_sampling_replays_exact_tokens() -> None:
    torch.manual_seed(4)
    model = DenseTransformer(model_config()).eval()
    prompt = torch.tensor([[1, 5, 6]])
    first_generator = torch.Generator().manual_seed(91)
    second_generator = torch.Generator().manual_seed(91)

    first = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.8,
        top_k=8,
        generator=first_generator,
    )
    second = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=0.8,
        top_k=8,
        generator=second_generator,
    )

    assert torch.equal(first, second)
