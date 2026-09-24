"""Chronological data splitting and batch sampling."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

import numpy as np
from torch.utils.data import Sampler

T = TypeVar("T")


def chronological_split(
    items: Sequence[T],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[T], list[T], list[T]]:
    """Split already chronological items without randomization."""
    if any(value < 0 for value in (train_ratio, validation_ratio, test_ratio)):
        raise ValueError("Split ratios must be nonnegative")
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to one")
    total = len(items)
    train_end = int(np.floor(total * train_ratio))
    validation_end = train_end + int(np.floor(total * validation_ratio))
    return (
        list(items[:train_end]),
        list(items[train_end:validation_end]),
        list(items[validation_end:]),
    )


class ChronologicalBatchSampler(Sampler[list[int]]):
    """Yield consecutive date indices; direction buffers are path-dependent."""

    def __init__(self, indices: Sequence[int], batch_size: int, drop_last: bool = False) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.indices = list(indices)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, len(self.indices), self.batch_size):
            batch = self.indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        quotient, remainder = divmod(len(self.indices), self.batch_size)
        return quotient if self.drop_last or remainder == 0 else quotient + 1


def chronological_blocks(length: int, number_of_blocks: int = 14) -> list[list[int]]:
    """Create consecutive blocks for paper-style time-series cross-validation."""
    if length < number_of_blocks or number_of_blocks < 2:
        raise ValueError("Need at least one observation per chronological block")
    return [list(chunk.astype(int)) for chunk in np.array_split(np.arange(length), number_of_blocks)]

