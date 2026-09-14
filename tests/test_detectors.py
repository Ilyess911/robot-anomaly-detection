"""The one-class detectors, and the sign convention that holds them together.

Every detector here must satisfy the same contract: fitted on healthy runs
only, it emits a score that is *higher* for anomalies. scikit-learn is
deliberately inconsistent about that direction, and an inverted detector still
produces plausible-looking output, so the convention is tested rather than
trusted.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from src.data.loader import encode_labels, load_robot_data, trace_ids
from src.evaluation.protocol import grouped_split
from src.features.statistical import create_statistical_features, feature_matrix
from src.models.detectors import DETECTORS, PCAReconstructionDetector


@pytest.fixture(scope="module")
def split():
    frame, _ = encode_labels(load_robot_data(), binary=True)
    X, _ = feature_matrix(create_statistical_features(frame))
    y = frame["label_encoded"].to_numpy(dtype=int)
    return grouped_split(X, y, trace_ids(frame))


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_a_detector_scores_failures_above_healthy_runs(name, split):
    """The sign convention, checked end to end on held-out data.

    The threshold is not involved: this is about the ranking. A detector whose
    ROC-AUC lands below 0.5 has its sign inverted, which is the one failure mode
    that produces output no visual inspection would catch.
    """
    detector = DETECTORS[name](random_state=42).fit(split.X_train[split.y_train == 0])
    scores = detector.anomaly_score(split.X_test)

    assert scores.shape == split.y_test.shape
    assert roc_auc_score(split.y_test, scores) > 0.9, f"{name} ranks failures below healthy runs"
    assert scores[split.y_test == 1].mean() > scores[split.y_test == 0].mean()


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_the_threshold_is_calibrated_on_healthy_training_scores(name, split):
    """A p95 cutoff must flag about 5% of the runs it was fitted on, not more.

    This is the property that makes the threshold rule usable without labels. If
    a detector alarmed on a third of its own training data, its false alarm rate
    in deployment would be unacceptable before it ever saw a real failure.
    """
    healthy = split.X_train[split.y_train == 0]
    detector = DETECTORS[name](random_state=42, percentile=95.0).fit(healthy)

    flagged = detector.predict(healthy).mean()
    assert flagged <= 0.06, f"{name} alarms on {flagged:.0%} of its own training data"


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_a_detector_refuses_to_score_before_it_is_fitted(name):
    with pytest.raises(RuntimeError):
        DETECTORS[name](random_state=42).anomaly_score(np.zeros((2, 48)))


def test_the_pca_detector_keeps_a_real_bottleneck(split):
    """A linear autoencoder that kept every component would reconstruct exactly.

    The reconstruction error is only informative because the projection loses
    something. With 48 features and 95% of the healthy variance retained, the
    subspace has to be strictly smaller than the input.
    """
    detector = PCAReconstructionDetector(random_state=42, variance=0.95)
    detector.fit(split.X_train[split.y_train == 0])
    assert 0 < detector.n_components < split.X_train.shape[1]
