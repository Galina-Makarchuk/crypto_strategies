"""Purged k-fold with embargo and walk-forward iterator.

Standard k-fold on time-series data leaks information across folds because
adjacent train and test samples share most of their feature lookback window.
The fix (de Prado, *Advances in Financial Machine Learning*, ch. 7) is:

1. Purge: remove from the training set any sample whose feature window
   overlaps the test set.
2. Embargo: also drop a buffer of bars immediately AFTER the test set, so
   the model can't be trained on bars whose label depends on the test
   period's price action.

Both helpers below operate on numpy index arrays and assume bars are sorted
ascending by time (the project's invariant). They never reorder data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedKFold:
    """K-fold splitter with purge + embargo for sequential data.

    Args:
        n_splits: number of folds.
        embargo: number of bars to drop from training data on either side
            of each test fold.
    """

    n_splits: int = 5
    embargo: int = 24

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if n_samples < self.n_splits * 4:
            raise ValueError(
                f"Need at least 4*n_splits ({4 * self.n_splits}) samples; got {n_samples}"
            )
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        bounds: list[tuple[int, int]] = []
        for size in fold_sizes:
            bounds.append((current, current + size))
            current += size

        all_idx = np.arange(n_samples)
        for start, stop in bounds:
            test_idx = all_idx[start:stop]
            train_mask = np.ones(n_samples, dtype=bool)
            embargo_lo = max(0, start - self.embargo)
            embargo_hi = min(n_samples, stop + self.embargo)
            train_mask[embargo_lo:embargo_hi] = False
            train_idx = all_idx[train_mask]
            yield train_idx, test_idx


def walk_forward_iter(
    n_samples: int,
    train_size: int,
    test_size: int,
    embargo: int = 24,
    step: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Anchored or rolling walk-forward splits — chronological train/test.

    Each fold's train block precedes its test block in time, separated by
    ``embargo`` bars. Use this for the *final* OOS evaluation; ``PurgedKFold``
    is for hyperparameter tuning where you want full data coverage.

    If ``step`` is None it defaults to ``test_size`` (non-overlapping tests).
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    cursor = train_size + embargo
    while cursor + test_size <= n_samples:
        train_end = cursor - embargo
        train_start = max(0, train_end - train_size)
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(cursor, cursor + test_size)
        yield train_idx, test_idx
        cursor += step
