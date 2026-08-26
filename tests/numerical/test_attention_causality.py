from __future__ import annotations

import torch
from conftest import model_config

from aurora.model.attention import GroupedQueryAttention


def test_attention_output_cannot_depend_on_future_tokens() -> None:
    torch.manual_seed(0)
    module = GroupedQueryAttention(model_config(dropout=0.0)).eval()
    original = torch.randn(1, 6, 32)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100.0

    original_output = module(original)
    changed_output = module(changed)

    torch.testing.assert_close(
        original_output[:, :4],
        changed_output[:, :4],
        rtol=0.0,
        atol=1e-6,
    )
