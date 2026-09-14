"""Lightweight one-class detectors, all trained on healthy executions only.

Why one-class, and why these four
---------------------------------

Training on healthy data alone mirrors deployment order. A new cell has months
of healthy operation and no catalogue of the failures it has not had yet, so a
supervised classifier is not available on day one and a one-class detector is.

The four here were chosen to span the ways a detector can be cheap, because the
research question this repository asks is about *lightweight* models:

=====================  ==================================================
Isolation Forest       partitioning, no distance, no distribution assumed
One-Class SVM          a kernel boundary around the healthy region
PCA reconstruction     a linear autoencoder: project, rebuild, measure error
Mahalanobis            one shrunk covariance, the classical control chart
=====================  ==================================================

A nonlinear autoencoder is deliberately absent. There are 103 healthy
executions in the training half and 48 features; a network with a hidden layer
wide enough to be called one would carry more parameters than it has samples to
fit them, and its reconstruction error would say more about initialisation than
about the robot. PCA reconstruction error is the linear case of the same idea,
it is honest at this sample size, and it is listed among the extensions that a
larger dataset would make worth revisiting.

Sign convention
---------------

Every detector exposes ``anomaly_score`` where **higher means more anomalous**.
scikit-learn is inconsistent about this on purpose (its ``score_samples`` grows
with normality), and reconciling it at the boundary is what stops a sign error
from turning a good detector into an inverted one that still looks plausible.

Thresholds
----------

``predict`` needs a cutoff, and the only rule available without labelled
failures is a percentile of the score distribution on the healthy training
runs. That is what ``fit`` stores and what ``predict`` uses by default. The
best threshold an oracle could have picked is reported separately in the
experiments, and the gap between the two is the honest cost of not having
labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

#: Percentile of the healthy training scores used as the default cutoff. At 95
#: the detector accepts a 5% false alarm rate on data it has already seen,
#: which is the usual starting point on a line where an alarm costs a stop.
DEFAULT_PERCENTILE = 95.0


@dataclass
class BaseDetector:
    """Shared plumbing: scale on healthy data, then calibrate a threshold.

    Subclasses implement ``_fit_core`` and ``_score_core``. Everything else,
    including the sign convention and the percentile threshold, is handled here
    so that four detectors cannot drift apart in the details that matter.
    """

    name: str = "detector"
    percentile: float = DEFAULT_PERCENTILE
    random_state: int = 42
    scaler: StandardScaler | None = field(default=None, repr=False)
    threshold_: float = field(default=float("nan"), repr=False)
    train_scores_: np.ndarray | None = field(default=None, repr=False)

    def _fit_core(self, X: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _score_core(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def fit(self, X_healthy: np.ndarray) -> BaseDetector:
        """Fit on healthy executions only, then calibrate the cutoff on them."""
        self.scaler = StandardScaler().fit(X_healthy)
        self._fit_core(self.scaler.transform(X_healthy))
        self.train_scores_ = self._score_core(self.scaler.transform(X_healthy))
        self.threshold_ = float(np.percentile(self.train_scores_, self.percentile))
        logger.info(
            "%s fitted on %d healthy runs, threshold p%.0f = %.4f",
            self.name,
            len(X_healthy),
            self.percentile,
            self.threshold_,
        )
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Continuous anomaly score, higher meaning more anomalous."""
        if self.scaler is None:
            raise RuntimeError(f"{self.name} is not fitted")
        return self._score_core(self.scaler.transform(X))

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """1 for a flagged execution, 0 otherwise."""
        cutoff = self.threshold_ if threshold is None else threshold
        return (self.anomaly_score(X) > cutoff).astype(int)


@dataclass
class IsolationForestDetector(BaseDetector):
    """Anomalies are the points a random partitioning isolates in few splits.

    No distance, no distributional assumption, and a cost that stays flat as the
    feature count grows, which is why it is the usual first detector on an
    industrial signal.
    """

    name: str = "isolation_forest"
    n_estimators: int = 200
    model: IsolationForest | None = field(default=None, repr=False)

    def _fit_core(self, X: np.ndarray) -> None:
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X)

    def _score_core(self, X: np.ndarray) -> np.ndarray:
        # score_samples grows with normality, so it is negated to satisfy the
        # convention that higher means more anomalous.
        return -self.model.score_samples(X)


@dataclass
class OneClassSVMDetector(BaseDetector):
    """A kernel boundary drawn around the healthy region.

    ``nu`` is an upper bound on the fraction of training points allowed outside
    that boundary. It is a modelling choice, not a fit to the test set, and the
    threshold percentile is calibrated on top of the resulting score anyway.
    """

    name: str = "one_class_svm"
    nu: float = 0.1
    gamma: str | float = "scale"
    model: OneClassSVM | None = field(default=None, repr=False)

    def _fit_core(self, X: np.ndarray) -> None:
        self.model = OneClassSVM(kernel="rbf", gamma=self.gamma, nu=self.nu).fit(X)

    def _score_core(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


@dataclass
class PCAReconstructionDetector(BaseDetector):
    """Project onto the healthy subspace, rebuild, and measure what is missing.

    This is the linear autoencoder. The components are fitted on healthy runs,
    so a healthy execution is rebuilt almost exactly and a failure, whose
    variation lies outside that subspace, is not. The residual is the score.

    ``n_components`` is set by explained variance rather than by a fixed count,
    so the bottleneck adapts to how much structure the healthy data actually
    has instead of to a number chosen for its roundness.
    """

    name: str = "pca_reconstruction"
    variance: float = 0.95
    model: PCA | None = field(default=None, repr=False)

    def _fit_core(self, X: np.ndarray) -> None:
        self.model = PCA(n_components=self.variance, svd_solver="full").fit(X)
        logger.info(
            "%s kept %d components for %.0f%% of healthy variance",
            self.name,
            self.model.n_components_,
            self.variance * 100,
        )

    def _score_core(self, X: np.ndarray) -> np.ndarray:
        rebuilt = self.model.inverse_transform(self.model.transform(X))
        return np.sum((X - rebuilt) ** 2, axis=1)

    @property
    def n_components(self) -> int:
        return int(self.model.n_components_) if self.model is not None else 0


@dataclass
class MahalanobisDetector(BaseDetector):
    """Distance to the healthy mean, in units of the healthy covariance.

    The classical multivariate control chart, and the baseline that every newer
    detector should be made to beat. Ledoit-Wolf shrinkage is not decoration:
    the covariance is estimated from 103 executions in 48 dimensions, where the
    empirical estimate is singular and its inverse is noise.
    """

    name: str = "mahalanobis"
    estimator: LedoitWolf | None = field(default=None, repr=False)

    def _fit_core(self, X: np.ndarray) -> None:
        self.estimator = LedoitWolf().fit(X)

    def _score_core(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.mahalanobis(X)


#: The registry the experiments iterate over. Adding a detector here is enough
#: for it to appear in every table, figure and report the project publishes.
DETECTORS = {
    "isolation_forest": IsolationForestDetector,
    "one_class_svm": OneClassSVMDetector,
    "pca_reconstruction": PCAReconstructionDetector,
    "mahalanobis": MahalanobisDetector,
}


def build_detectors(random_state: int = 42, percentile: float = DEFAULT_PERCENTILE) -> list:
    """One fresh instance of every registered detector."""
    return [
        factory(random_state=random_state, percentile=percentile) for factory in DETECTORS.values()
    ]
