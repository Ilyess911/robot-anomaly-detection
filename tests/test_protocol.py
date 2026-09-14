"""Does the evaluation protocol keep its promises.

The README publishes numbers and a conclusion. These tests check that the
protocol producing them does not drift: the baseline, the size of the test set,
and the fact that both families of models are judged on the same sample. Those
are the three things whose falseness would make every score incomparable
without any single one of them looking suspicious.
"""

from __future__ import annotations

import numpy as np
from scripts.benchmark import RANDOM_STATE, TEST_SIZE, build_dataset, run_baselines, score
from sklearn.model_selection import train_test_split

N_TEST = 93
N_TRAIN = 370


def split():
    X, y = build_dataset()
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def test_the_split_is_the_one_announced():
    X_train, X_test, y_train, y_test = split()
    assert len(y_train) == N_TRAIN
    assert len(y_test) == N_TEST
    assert X_train.shape[1] == X_test.shape[1] == 48


def test_stratification_preserves_the_imbalance():
    _, _, y_train, y_test = split()
    assert abs((y_train == 1).mean() - (y_test == 1).mean()) < 0.01


def test_the_constant_baseline_is_the_one_in_the_readme():
    """0.838 F1 without learning anything. That figure gives every other one its
    meaning, and it is the only one the README cannot afford to get wrong."""
    _, _, _, y_test = split()
    baselines = run_baselines(y_test)
    assert round(baselines["always_anomaly"]["f1_anomaly"], 3) == 0.838
    assert round(baselines["always_anomaly"]["accuracy"], 3) == 0.720
    assert baselines["always_normal"]["f1_anomaly"] == 0.0


def test_a_model_on_shuffled_labels_cannot_beat_the_constant_answer():
    """The control that was missing, and that makes a perfect score credible.

    Random Forest reaches 1.0000 on the held-out set. On 463 executions a
    perfect score is first of all a warning, and it has to be shown to come from
    the data rather than from a leak.

    If the pipeline leaked, the model would stay good even on labels drawn at
    random. Trained on noise it must instead fall *below* the constant answer,
    because it fits patterns that are not there.

    This control is necessary and it is not sufficient. It passed for a year
    while 46% of the dataset was duplicated executions, because a permutation
    breaks the association the duplicates carry. ``test_duplicates.py`` is what
    catches that one.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y = build_dataset()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    forest = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )

    real = cross_val_score(forest, X, y, cv=cv, scoring="f1").mean()
    shuffled = cross_val_score(
        forest, X, np.random.default_rng(0).permutation(y), cv=cv, scoring="f1"
    ).mean()

    assert real > 0.95, "the real labels must stay learnable"
    assert shuffled < 0.83, f"shuffled labels at {shuffled:.4f}: the pipeline leaks"


def test_the_reference_f1_is_the_anomaly_class():
    """The notebooks' central defect: three definitions of F1 in three places.

    Here ``f1_anomaly`` is the positive class and ``f1_weighted`` the weighted
    average. On an imbalanced set the two differ widely, and this test fails if
    anyone reunifies them by accident.
    """
    y_true = np.array([1] * 71 + [0] * 22)
    y_pred = np.ones(93, dtype=int)
    result = score(y_true, y_pred, None)
    assert result["f1_anomaly"] > result["f1_weighted"]
    assert result["recall_anomaly"] == 1.0


def test_the_features_do_not_contain_the_target():
    """A trivial model must stay away from perfection.

    If a target column slipped into X, a logistic regression would reach 1.0000.
    The bar is deliberately high: this does not test model quality, it tests the
    absence of a leak.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = split()
    model = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
    ).fit(X_train, y_train)
    assert model.score(X_test, y_test) < 0.999
