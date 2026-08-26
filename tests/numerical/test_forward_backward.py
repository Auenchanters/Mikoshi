from __future__ import annotations

import torch
from conftest import model_config

from aurora.model.transformer import DenseTransformer


def test_forward_backward_is_finite_and_reaches_all_trainable_parameters() -> None:
    torch.manual_seed(5)
    model = DenseTransformer(model_config(dropout=0.0))
    input_ids = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))

    loss = model(input_ids, targets=targets).loss

    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
