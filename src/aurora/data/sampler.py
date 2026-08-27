from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import cast

import torch
from torch.utils.data import Sampler


class DeterministicBatchSampler(Sampler[list[int]]):
    """Seeded epoch permutations with a checkpointable next-batch cursor."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        if drop_last and dataset_size < batch_size:
            raise ValueError("drop_last sampler requires dataset_size >= batch_size")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.cursor = 0
        self.order = self._permutation(self.epoch)

    def _permutation(self, epoch: int) -> list[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        return cast(
            list[int],
            torch.randperm(self.dataset_size, generator=generator).tolist(),
        )

    def _limit(self) -> int:
        if self.drop_last:
            return self.dataset_size - (self.dataset_size % self.batch_size)
        return self.dataset_size

    def __iter__(self) -> Iterator[list[int]]:
        limit = self._limit()
        while self.cursor < limit:
            end = min(self.cursor + self.batch_size, limit)
            batch = self.order[self.cursor : end]
            self.cursor = end
            yield batch
        self.epoch += 1
        self.cursor = 0
        self.order = self._permutation(self.epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def state_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "cursor": self.cursor,
            "dataset_size": self.dataset_size,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "order": list(self.order),
            "seed": self.seed,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected_keys = {
            "batch_size",
            "cursor",
            "dataset_size",
            "drop_last",
            "epoch",
            "order",
            "seed",
        }
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            unknown = sorted(set(state) - expected_keys)
            raise ValueError(f"invalid sampler state keys; missing={missing}, unknown={unknown}")
        for name, current in (
            ("dataset_size", self.dataset_size),
            ("batch_size", self.batch_size),
            ("seed", self.seed),
            ("drop_last", self.drop_last),
        ):
            if state[name] != current:
                raise ValueError(f"sampler {name} does not match checkpoint state")
        epoch = state["epoch"]
        cursor = state["cursor"]
        order = state["order"]
        if type(epoch) is not int or cast(int, epoch) < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")
        if type(cursor) is not int or not 0 <= cast(int, cursor) <= self._limit():
            raise ValueError("sampler cursor is outside the epoch range")
        if cast(int, cursor) % self.batch_size != 0:
            raise ValueError("sampler cursor must be aligned to a completed batch")
        if not isinstance(order, list) or any(type(index) is not int for index in order):
            raise ValueError("sampler order must be a list of integer indices")
        typed_order = cast(list[int], order)
        if len(typed_order) != self.dataset_size or sorted(typed_order) != list(
            range(self.dataset_size)
        ):
            raise ValueError("sampler order must be a permutation of the dataset indices")
        if typed_order != self._permutation(cast(int, epoch)):
            raise ValueError("sampler order does not match its seed and epoch")
        self.epoch = cast(int, epoch)
        self.cursor = cast(int, cursor)
        self.order = list(typed_order)
