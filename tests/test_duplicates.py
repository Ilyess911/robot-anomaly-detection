"""The duplicate finding, guarded the way the labelling finding is.

The five subsets are not five recordings. LP2 and LP3 annotate the same 47
executions under two fault taxonomies, LP4 and LP5 share 116 more, and the
merged frame holds 463 rows over 251 distinct traces. Under a random split, two
thirds of the test set has an exact copy in the training half.

These tests exist so that the finding cannot quietly disappear. If a future
change to the parser, the feature code or the split makes these numbers move,
the README stops being true and the suite says so.
"""

from __future__ import annotations

import numpy as np
from src.data.loader import duplicate_summary, encode_labels, load_robot_data, trace_ids
from src.evaluation.protocol import grouped_split, random_split
from src.features.statistical import create_statistical_features, feature_matrix

N_EXECUTIONS = 463
N_DISTINCT_TRACES = 251
N_LEAKED_UNDER_RANDOM_SPLIT = 64


def dataset():
    frame, _ = encode_labels(load_robot_data(), binary=True)
    features = create_statistical_features(frame)
    X, _ = feature_matrix(features)
    return X, frame["label_encoded"].to_numpy(dtype=int), trace_ids(frame), frame


def test_the_dataset_holds_fewer_traces_than_rows():
    """463 executions, 251 distinct recordings. The gap is the whole finding."""
    _, _, _, frame = dataset()
    summary = duplicate_summary(frame)
    assert summary["n_executions"] == N_EXECUTIONS
    assert summary["n_distinct_traces"] == N_DISTINCT_TRACES
    assert 0.45 < summary["duplicate_share"] < 0.46


def test_duplicated_traces_never_disagree_on_the_binary_label():
    """The taxonomies differ between subsets; healthy against failed does not.

    This is what makes the merge defensible at all. LP2 calls an execution
    ``left_col`` and LP3 calls the same recording ``moved``, but neither ever
    calls it healthy while the other calls it a failure. Were that to happen,
    the binary target itself would be ambiguous and no score would mean
    anything.
    """
    _, _, _, frame = dataset()
    assert duplicate_summary(frame)["traces_with_conflicting_binary_label"] == 0


def test_the_random_split_hands_the_model_most_of_its_test_set():
    """The number that explains the perfect scores this project used to report."""
    X, y, groups, _ = dataset()
    split = random_split(X, y, groups)
    assert split.leaked_test_rows == N_LEAKED_UNDER_RANDOM_SPLIT
    assert split.leaked_test_share > 0.65


def test_the_grouped_split_leaks_nothing_and_keeps_the_balance():
    """Same sizes, same class balance, no trace on both sides.

    The comparison in the README is only fair because these three properties
    hold: if the grouped split were smaller or differently balanced, the gap
    between the two protocols would confound the grouping with something else.
    """
    X, y, groups, _ = dataset()
    random = random_split(X, y, groups)
    grouped = grouped_split(X, y, groups)

    assert grouped.leaked_test_rows == 0
    assert len(grouped.y_test) == len(random.y_test)
    assert abs(grouped.y_test.mean() - random.y_test.mean()) < 0.01


def test_grouping_costs_the_forest_its_perfect_cross_validation():
    """The duplicates do not create the performance. They erase the variance.

    A random forest cross-validated without grouping reports 1.0000 with a
    standard deviation of exactly zero, which on 463 samples is the signature of
    a leak rather than of a good model. Under grouped folds the score falls and
    the variance appears. The assertion is on that direction, not on an exact
    value, so it survives a library upgrade.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score

    X, y, groups, _ = dataset()
    forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    ungrouped = cross_val_score(
        forest, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1"
    )
    grouped = cross_val_score(
        forest,
        X,
        y,
        groups=groups,
        cv=StratifiedGroupKFold(5, shuffle=True, random_state=42),
        scoring="f1",
    )

    assert ungrouped.mean() > grouped.mean(), "grouping must not flatter the model"
    assert np.isclose(ungrouped.std(), 0.0, atol=0.01)
    assert grouped.std() > ungrouped.std(), "the honest protocol must show variance"
