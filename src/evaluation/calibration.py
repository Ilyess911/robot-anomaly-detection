"""Setting an alarm threshold when no failure has ever been labelled.

The problem, stated precisely
-----------------------------

A one-class detector produces a score. Turning that score into an alarm needs a
threshold, and the threshold is the only part of the system that a deployment
cannot copy from the training data's labels, because there are none. Every
result in this repository says the same thing: on this data the ranking is
solved and the threshold is not.

Two rules are compared here, and they differ in what they promise.

``percentile``
    Put the threshold at the 95th percentile of the healthy training scores.
    This is what almost every tutorial does. It promises nothing: the 95th
    percentile of 103 samples is an *estimate* of the 95th percentile of the
    distribution, and an estimate computed from the upper tail of a small sample
    is a poor one. The realised false alarm rate can sit well above 5%.

``tolerance``
    Put the threshold at the k-th largest healthy training score, with k chosen
    so that the false alarm rate stays below a target with stated confidence.
    This is a distribution-free one-sided tolerance bound and it is exact: for
    i.i.d. samples, the probability that a new healthy execution exceeds the
    k-th largest of n follows Beta(k, n - k + 1), whatever the underlying
    distribution. Requiring the (1 - delta) quantile of that Beta to stay under
    the target gives a guarantee rather than an estimate.

The price is sensitivity. The guarantee forces a higher threshold than the naive
percentile, so recall falls. That trade is the honest content of this module:
a controlled false alarm rate costs detections, and the exchange rate is
measurable.

Why this matters on a line
--------------------------

A false alarm stops a cell. An operator who is stopped three times a shift for
nothing switches the detector off in a week, and then its recall is zero. A rule
that can state "at most 5% of healthy executions will be flagged, with 90%
confidence" is deployable; one that merely hopes so is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Threshold:
    """A threshold, the rule that produced it, and what that rule promises.

    Attributes:
        value: the score above which an execution is flagged.
        rule: ``percentile`` or ``tolerance``.
        target_far: the false alarm rate the rule was asked for.
        guaranteed_far: the upper bound the rule actually guarantees at
            ``confidence``. ``nan`` for rules that guarantee nothing.
        confidence: the probability with which the bound holds.
        order_statistic: which training score was used, counting from the top.
            1 is the largest. ``None`` for the percentile rule.
    """

    value: float
    rule: str
    target_far: float
    guaranteed_far: float
    confidence: float
    order_statistic: int | None = None

    def to_dict(self) -> dict:
        return {
            "value": float(self.value),
            "rule": self.rule,
            "target_far": self.target_far,
            "guaranteed_far": None if np.isnan(self.guaranteed_far) else float(self.guaranteed_far),
            "confidence": self.confidence,
            "order_statistic": self.order_statistic,
        }


def percentile_threshold(train_scores: np.ndarray, target_far: float = 0.05) -> Threshold:
    """The usual rule: the empirical quantile of the healthy training scores.

    Included as the baseline to beat, not as a recommendation. It guarantees
    nothing, which is stated rather than hidden: ``guaranteed_far`` is ``nan``.
    """
    value = float(np.percentile(train_scores, 100 * (1 - target_far)))
    return Threshold(
        value=value,
        rule="percentile",
        target_far=target_far,
        guaranteed_far=float("nan"),
        confidence=float("nan"),
    )


def tolerance_threshold(
    train_scores: np.ndarray, target_far: float = 0.05, confidence: float = 0.90
) -> Threshold:
    """The k-th largest training score, with k chosen to bound the false alarm rate.

    For n i.i.d. healthy scores, the probability that a new healthy execution
    exceeds the k-th largest is distributed Beta(k, n - k + 1). That fact holds
    for any continuous distribution, which is what makes the bound
    distribution-free and why no assumption about the score's shape appears
    anywhere in this file.

    The largest admissible k is taken, so the threshold is the most sensitive
    one the guarantee allows. A smaller k would be safer and blinder.

    Args:
        train_scores: anomaly scores on healthy training executions.
        target_far: the false alarm rate not to exceed.
        confidence: the probability with which the bound must hold.

    Returns:
        A :class:`Threshold`. If not even k = 1 satisfies the bound, the sample
        is too small for the requested guarantee; the largest training score is
        returned and the achieved bound is reported as it is, so the caller can
        see that the guarantee was not met rather than assume it was.
    """
    ordered = np.sort(np.asarray(train_scores, dtype=float))[::-1]
    n = len(ordered)

    best_k = None
    achieved = float("nan")
    for k in range(1, n + 1):
        bound = float(scipy_stats.beta.ppf(confidence, k, n - k + 1))
        if bound <= target_far:
            best_k, achieved = k, bound
        else:
            break

    if best_k is None:
        bound = float(scipy_stats.beta.ppf(confidence, 1, n))
        logger.warning(
            "%d healthy samples cannot guarantee a false alarm rate of %.1f%% at %.0f%% "
            "confidence; the tightest achievable bound is %.1f%%",
            n,
            target_far * 100,
            confidence * 100,
            bound * 100,
        )
        return Threshold(
            value=float(ordered[0]),
            rule="tolerance",
            target_far=target_far,
            guaranteed_far=bound,
            confidence=confidence,
            order_statistic=1,
        )

    return Threshold(
        value=float(ordered[best_k - 1]),
        rule="tolerance",
        target_far=target_far,
        guaranteed_far=achieved,
        confidence=confidence,
        order_statistic=best_k,
    )


def realised_far(threshold: Threshold, healthy_scores: np.ndarray) -> float:
    """The false alarm rate the threshold actually produces on unseen healthy runs.

    This is the number that decides whether a rule works. A rule whose target is
    5% and whose realised rate is 20% has not been conservative, it has been
    wrong.
    """
    if len(healthy_scores) == 0:
        return float("nan")
    return float((np.asarray(healthy_scores) > threshold.value).mean())


def calibration_gap(threshold: Threshold, healthy_scores: np.ndarray) -> dict:
    """Target, guarantee and outcome side by side, for one threshold."""
    realised = realised_far(threshold, healthy_scores)
    return {
        **threshold.to_dict(),
        "realised_far": realised,
        "n_healthy_evaluated": len(healthy_scores),
        "exceeds_target": bool(realised > threshold.target_far),
    }
