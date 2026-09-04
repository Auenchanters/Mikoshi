from __future__ import annotations

import pytest
import torch

from aurora.model.recurrent_stability import InitialRMSStabilizer


def _per_token_rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor.float(), dim=-1) / tensor.shape[-1] ** 0.5


def test_initial_rms_stabilizer_matches_each_initial_token_rms() -> None:
    stabilizer = InitialRMSStabilizer(norm_eps=1e-6)
    initial = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, -2.0, 2.0, -2.0]],
            [[0.5, -1.5, 2.5, -3.5], [4.0, 3.0, -2.0, -1.0]],
        ]
    )
    candidate = torch.tensor(
        [
            [[4.0, -3.0, 2.0, -1.0], [1.0, 3.0, 5.0, 7.0]],
            [[-6.0, 2.0, 1.0, 3.0], [0.25, -0.5, 0.75, -1.0]],
        ]
    )

    stabilized = stabilizer(initial, candidate)

    assert stabilized.dtype == candidate.dtype
    assert torch.allclose(_per_token_rms(stabilized), _per_token_rms(initial), atol=1e-6)


@pytest.mark.parametrize(
    ("initial", "candidate"),
    [
        (torch.zeros(1, 2, 4), torch.tensor([[[1.0, -2.0, 3.0, -4.0]]]).repeat(1, 2, 1)),
        (torch.ones(1, 2, 4), torch.zeros(1, 2, 4)),
        (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4)),
    ],
)
def test_initial_rms_stabilizer_handles_zero_initial_or_candidate(
    initial: torch.Tensor, candidate: torch.Tensor
) -> None:
    stabilized = InitialRMSStabilizer(norm_eps=1e-6)(initial, candidate)

    assert torch.isfinite(stabilized).all()
    assert torch.equal(stabilized, torch.zeros_like(stabilized))


def test_initial_rms_stabilizer_keeps_extreme_finite_candidate_finite() -> None:
    initial = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])
    candidate = torch.tensor([[[1.0e20, -1.0e20, 1.0e20, -1.0e20]]])

    stabilized = InitialRMSStabilizer(norm_eps=1e-6)(initial, candidate)

    assert torch.isfinite(stabilized).all()
    assert torch.allclose(_per_token_rms(stabilized), _per_token_rms(initial), atol=1e-6)


def test_initial_rms_stabilizer_preserves_subnormal_initial_rms() -> None:
    component = torch.finfo(torch.float32).tiny / 2
    initial = torch.full((1, 1, 4), component)
    candidate = torch.ones_like(initial)

    stabilized = InitialRMSStabilizer(norm_eps=1e-6)(initial, candidate)

    assert torch.equal(stabilized, initial)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    ("initial_kind", "candidate_kind"),
    [("maximum", "zero"), ("zero", "maximum")],
)
def test_initial_rms_stabilizer_keeps_combined_zero_and_extreme_inputs_finite(
    dtype: torch.dtype, initial_kind: str, candidate_kind: str
) -> None:
    maximum = torch.finfo(dtype).max
    initial = (
        torch.full((1, 1, 4), maximum, dtype=dtype)
        if initial_kind == "maximum"
        else torch.zeros((1, 1, 4), dtype=dtype)
    )
    candidate = (
        torch.full((1, 1, 4), maximum, dtype=dtype)
        if candidate_kind == "maximum"
        else torch.zeros((1, 1, 4), dtype=dtype)
    )

    stabilized = InitialRMSStabilizer(norm_eps=1e-6)(initial, candidate)

    assert stabilized.dtype == dtype
    assert torch.isfinite(stabilized).all()
    assert torch.equal(stabilized, torch.zeros_like(stabilized))


def test_initial_rms_stabilizer_has_finite_candidate_gradient_at_zero() -> None:
    initial = torch.ones((1, 1, 4))
    candidate = torch.zeros((1, 1, 4), requires_grad=True)

    stabilized = InitialRMSStabilizer(norm_eps=1e-6)(initial, candidate)
    stabilized.sum().backward()

    assert torch.isfinite(stabilized).all()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert float(candidate.grad.abs().sum()) > 0.0


def test_initial_rms_stabilizer_has_no_parameters_or_persistent_state() -> None:
    stabilizer = InitialRMSStabilizer(norm_eps=1e-6)

    assert list(stabilizer.parameters()) == []
    assert stabilizer.state_dict() == {}
