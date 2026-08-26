from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from aurora.model.recurrent import FixedLoopRecurrentTransformer
from tests.recurrent_helpers import recurrent_model_config


def _assert_module_has_finite_nonzero_gradients(module: nn.Module) -> None:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_one_iteration_exactly_matches_explicit_prelude_core_coda() -> None:
    torch.manual_seed(17)
    model = FixedLoopRecurrentTransformer(recurrent_model_config(num_iterations=1)).cpu().eval()
    input_ids = torch.tensor([[1, 4, 7, 3, 9, 2]], dtype=torch.long)
    targets = torch.tensor([[4, 7, 3, 9, 2, 8]], dtype=torch.long)

    recurrent = model(input_ids, targets=targets)
    hidden = model.embedding_dropout(model.token_embedding(input_ids))
    hidden = model.prelude(hidden)
    hidden = model.core(hidden)
    hidden = model.coda(hidden)
    explicit_logits = model.lm_head(model.final_norm(hidden))
    explicit_loss = functional.cross_entropy(
        explicit_logits.float().reshape(-1, model.config.vocab_size),
        targets.reshape(-1),
    )

    assert torch.equal(recurrent.logits, explicit_logits)
    assert recurrent.loss is not None
    assert torch.equal(recurrent.loss, explicit_loss)

    recurrent.loss.backward()
    _assert_module_has_finite_nonzero_gradients(model.prelude)
    _assert_module_has_finite_nonzero_gradients(model.core)
    _assert_module_has_finite_nonzero_gradients(model.coda)
