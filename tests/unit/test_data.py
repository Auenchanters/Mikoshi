from __future__ import annotations

import pytest
import torch

from aurora.data.dataset import TokenBlockDataset
from aurora.data.sampler import DeterministicBatchSampler


def test_token_blocks_shift_targets_by_one_token() -> None:
    dataset = TokenBlockDataset(list(range(10)), sequence_length=4, stride=4)

    inputs, targets = dataset[1]

    assert inputs.tolist() == [4, 5, 6, 7]
    assert targets.tolist() == [5, 6, 7, 8]
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_token_blocks_reject_stream_without_one_complete_shifted_window() -> None:
    with pytest.raises(ValueError, match=r"at least sequence_length \+ 1"):
        TokenBlockDataset([1, 2, 3, 4], sequence_length=4)


def test_sampler_seed_controls_order_deterministically() -> None:
    first = list(DeterministicBatchSampler(12, batch_size=3, seed=7))
    second = list(DeterministicBatchSampler(12, batch_size=3, seed=7))
    changed = list(DeterministicBatchSampler(12, batch_size=3, seed=8))

    assert first == second
    assert first != changed
    assert sorted(index for batch in first for index in batch) == list(range(12))


def test_sampler_resume_returns_exact_next_batches() -> None:
    sampler = DeterministicBatchSampler(12, batch_size=3, seed=7)
    iterator = iter(sampler)
    consumed = [next(iterator), next(iterator)]
    state = sampler.state_dict()
    expected_tail = list(iterator)

    resumed = DeterministicBatchSampler(12, batch_size=3, seed=7)
    resumed.load_state_dict(state)

    assert len(consumed) == 2
    assert list(resumed) == expected_tail


def test_sampler_rejects_state_from_different_dataset() -> None:
    state = DeterministicBatchSampler(12, batch_size=3, seed=7).state_dict()
    incompatible = DeterministicBatchSampler(15, batch_size=3, seed=7)

    with pytest.raises(ValueError, match="dataset_size"):
        incompatible.load_state_dict(state)
