"""Is the dataset the one the README describes.

These tests do not check that the code runs. They check that the published
claims stay true. A README announcing 463 executions next to a parser that
returns 460 after a refactor is an involuntary lie, and it is exactly the kind
nobody notices.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.data.loader import encode_labels, load_robot_data
from src.features.statistical import create_statistical_features

# The README's numbers. Changing them here forces changing them there.
N_EXECUTIONS = 463
N_RAW_FEATURES = 90
N_STAT_FEATURES = 48
N_ORIGINAL_LABELS = 16
N_HEALTHY = 129  # 'normal' (109) plus 'ok' (20)
SUBSET_SIZES = {"LP1": 88, "LP2": 47, "LP3": 47, "LP4": 117, "LP5": 164}


@pytest.fixture(scope="module")
def frame():
    return load_robot_data()


def test_every_execution_is_loaded(frame):
    assert len(frame) == N_EXECUTIONS


def test_each_subset_keeps_its_size(frame):
    assert frame["source"].value_counts().to_dict() == SUBSET_SIZES


def test_an_execution_is_six_sensors_over_fifteen_steps(frame):
    """90 values per execution and not one more: 6 channels x 15 readings."""
    features = [column for column in frame.columns if column.startswith("feature_")]
    assert len(features) == N_RAW_FEATURES
    assert frame[features].isna().sum().sum() == 0


def test_the_published_imbalance_is_the_right_one(frame):
    """The README says 72% failures. That figure carries the reading of every
    score: if it moves, the baseline moves and the results change meaning."""
    encoded, _ = encode_labels(frame.copy(), binary=True)
    healthy = int((encoded["label_binary"] == 0).sum())
    assert healthy == N_HEALTHY
    assert frame["label"].nunique() == N_ORIGINAL_LABELS
    assert 0.71 < 1 - healthy / len(frame) < 0.73


def test_there_are_exactly_forty_eight_statistical_features(frame):
    encoded, _ = encode_labels(frame.copy(), binary=True)
    stats = create_statistical_features(encoded)
    meta = {"label", "label_encoded", "label_binary", "label_original", "source"}
    assert len([c for c in stats.columns if c not in meta]) == N_STAT_FEATURES


def test_no_statistical_feature_reproduces_the_target(frame):
    """The trap that returned an F1 of 1.0000 while the benchmark was written.

    ``create_statistical_features`` returns ``label_encoded`` and
    ``label_binary`` alongside the features. Both are numeric, so a selection by
    dtype carries them in and the model learns the answer. This test fails if a
    feature column ever becomes perfectly correlated with the target.
    """
    encoded, _ = encode_labels(frame.copy(), binary=True)
    stats = create_statistical_features(encoded)
    meta = {"label", "label_encoded", "label_binary", "label_original", "source"}
    target = stats["label_binary"].to_numpy(dtype=float)

    for column in stats.columns:
        if column in meta:
            continue
        values = stats[column].to_numpy(dtype=float)
        if np.std(values) == 0:
            continue
        assert abs(np.corrcoef(values, target)[0, 1]) < 0.999, f"{column} reproduces the target"


def test_the_two_target_columns_agree(frame):
    """The notebooks use ``label_binary`` and the benchmark ``label_encoded``.
    Were they to diverge, the two would no longer measure the same thing."""
    encoded, _ = encode_labels(frame.copy(), binary=True)
    stats = create_statistical_features(encoded)
    assert (stats["label_encoded"].to_numpy() == stats["label_binary"].to_numpy()).all()
