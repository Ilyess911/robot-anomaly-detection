"""Splitting rules, and the difference between two of them.

The dataset holds 463 executions over 251 distinct sensor traces: LP2 and LP3
annotate the same 47 recordings under two fault taxonomies, and LP4 and LP5
share 116 more. A random 80/20 split therefore hands the model 69% of its test
set to memorise beforehand.

Two protocols are defined here and both are reported everywhere:

``random``
    The protocol every published result on this dataset uses, including this
    project's own earlier numbers. Kept so the cost of it can be measured.

``grouped``
    Every copy of a trace stays on one side of the split. This is the protocol
    the published headline numbers now use.

The gap between them is not an accusation. It is a measurement, and it is the
only way to say how much of a score was recall rather than generalisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Split:
    """One train/test partition, with the bookkeeping needed to audit it."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    groups_test: np.ndarray
    name: str

    @property
    def leaked_test_rows(self) -> int:
        """Test executions whose trace also appears in the training half."""
        return int(np.isin(self.groups_test, np.unique(self.groups_train)).sum())

    @property
    def leaked_test_share(self) -> float:
        """The same count as a share of the test set."""
        return self.leaked_test_rows / max(len(self.y_test), 1)

    def summary(self) -> dict:
        return {
            "protocol": self.name,
            "n_train": len(self.y_train),
            "n_test": len(self.y_test),
            "anomaly_share_train": float(self.y_train.mean()),
            "anomaly_share_test": float(self.y_test.mean()),
            "leaked_test_rows": self.leaked_test_rows,
            "leaked_test_share": float(self.leaked_test_share),
        }


def random_split(X, y, groups, test_size: float = 0.2, random_state: int = 42) -> Split:
    """Stratified random split, ignoring that copies of a trace exist."""
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=test_size, random_state=random_state, stratify=y
    )
    split = Split(X_train, X_test, y_train, y_test, g_train, g_test, "random")
    logger.info(
        "random split: %d train, %d test, %d test rows also in train",
        len(y_train),
        len(y_test),
        split.leaked_test_rows,
    )
    return split


def grouped_split(X, y, groups, n_splits: int = 5, random_state: int = 42) -> Split:
    """Stratified split where no trace appears on both sides.

    The first fold of a stratified grouped k-fold is taken as the test set,
    which gives roughly the same held-out proportion as ``test_size=1/n_splits``
    while keeping the class balance and the grouping constraint together. It is
    deterministic under a fixed seed, which a group shuffle is not in the same
    way.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_index, test_index = next(splitter.split(X, y, groups=groups))
    split = Split(
        X[train_index],
        X[test_index],
        y[train_index],
        y[test_index],
        groups[train_index],
        groups[test_index],
        "grouped",
    )
    logger.info(
        "grouped split: %d train, %d test, %d test rows also in train",
        len(split.y_train),
        len(split.y_test),
        split.leaked_test_rows,
    )
    return split


def cv_splitter(grouped: bool, n_splits: int = 5, random_state: int = 42):
    """The cross-validator matching a protocol.

    A grouped split evaluated with an ungrouped cross-validation would put the
    duplicates back inside the search, so the two choices have to travel
    together.
    """
    if grouped:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
