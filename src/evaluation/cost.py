"""What a detector costs on a line, rather than what it scores on a test set.

Why F1 is the wrong last word
-----------------------------

F1 weighs a missed failure and a false alarm equally. No maintenance operation
does. A missed collision damages tooling and can propagate down the line; a
false alarm stops a cell for a few minutes. The ratio between those two is a
property of the plant, not of the model, and it decides where the threshold
belongs.

This dataset also has the balance backwards. 72% of its executions are failures.
A production line sees the opposite, often by two orders of magnitude, and that
inversion is exactly the regime where precision on the rare class collapses.
Rather than list this as a limitation and stop, the analysis here treats the
failure rate as a free parameter and asks what the detector would be worth
across the range a real line might sit in.

The model
---------

For a threshold t, with false positive rate FPR(t) and true positive rate
TPR(t) measured on held-out data, the expected cost per execution is

    C(t) = pi * (1 - TPR(t)) * c_miss + (1 - pi) * FPR(t) * c_alarm

where pi is the failure rate of the line. Dividing by ``c_alarm`` leaves a
single free number, the cost ratio r = c_miss / c_alarm, so the analysis needs
no currency and no invented figures:

    C(t) / c_alarm = pi * (1 - TPR(t)) * r + (1 - pi) * FPR(t)

Two references bound the useful range. Doing nothing costs ``pi * r``. Stopping
on every execution costs ``1 - pi``. A detector earns its place only when its
best operating point sits below both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)

#: Failure rates spanning what a line might plausibly show, from a well-run cell
#: to one in trouble. The dataset's own 72% is deliberately outside this range.
PREVALENCES = (0.001, 0.005, 0.01, 0.05, 0.10)

#: Cost of a missed failure expressed in false alarms. 1 means indifference,
#: which no plant believes; 1000 means a missed failure is worth a thousand
#: unnecessary stops, which is the order of magnitude for tooling damage.
COST_RATIOS = (1, 3, 10, 30, 100, 300, 1000)


@dataclass(frozen=True)
class OperatingPoint:
    """The best threshold under one set of economic assumptions."""

    prevalence: float
    cost_ratio: float
    threshold: float
    expected_cost: float
    cost_of_doing_nothing: float
    cost_of_always_stopping: float
    false_positive_rate: float
    true_positive_rate: float

    @property
    def worth_deploying(self) -> bool:
        """Does the detector beat both trivial policies under these assumptions."""
        return self.expected_cost < min(self.cost_of_doing_nothing, self.cost_of_always_stopping)

    def to_dict(self) -> dict:
        return {
            "prevalence": self.prevalence,
            "cost_ratio": self.cost_ratio,
            "threshold": self.threshold,
            "expected_cost": self.expected_cost,
            "cost_of_doing_nothing": self.cost_of_doing_nothing,
            "cost_of_always_stopping": self.cost_of_always_stopping,
            "false_positive_rate": self.false_positive_rate,
            "true_positive_rate": self.true_positive_rate,
            "worth_deploying": self.worth_deploying,
            "saving_against_best_trivial": (
                min(self.cost_of_doing_nothing, self.cost_of_always_stopping) - self.expected_cost
            ),
        }


def best_operating_point(
    y_true: np.ndarray, scores: np.ndarray, prevalence: float, cost_ratio: float
) -> OperatingPoint:
    """Minimise expected cost over every threshold the ROC curve offers.

    The search runs over the ROC curve's own thresholds rather than a grid, so
    no operating point that the data can distinguish is missed and none that it
    cannot is invented.

    Note what is held fixed and what is not. TPR and FPR come from the held-out
    executions of this dataset; the failure rate ``prevalence`` is imposed from
    outside. That separation is deliberate: the detector's discrimination is
    measured, the line's economics are assumed, and the two are never mixed into
    one number without saying so.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    cost = prevalence * (1 - tpr) * cost_ratio + (1 - prevalence) * fpr
    best = int(np.argmin(cost))

    return OperatingPoint(
        prevalence=prevalence,
        cost_ratio=cost_ratio,
        threshold=float(thresholds[best]),
        expected_cost=float(cost[best]),
        cost_of_doing_nothing=float(prevalence * cost_ratio),
        cost_of_always_stopping=float(1 - prevalence),
        false_positive_rate=float(fpr[best]),
        true_positive_rate=float(tpr[best]),
    )


def cost_grid(
    y_true: np.ndarray,
    scores: np.ndarray,
    prevalences: tuple[float, ...] = PREVALENCES,
    cost_ratios: tuple[float, ...] = COST_RATIOS,
) -> list[dict]:
    """The whole economic surface, one row per assumption pair."""
    return [
        best_operating_point(y_true, scores, prevalence, ratio).to_dict()
        for prevalence in prevalences
        for ratio in cost_ratios
    ]


def break_even_cost_ratio(
    y_true: np.ndarray, scores: np.ndarray, prevalence: float, ratios=COST_RATIOS
) -> float | None:
    """The cheapest missed failure that still justifies running the detector.

    Below this ratio, the plant is better off ignoring the alarms entirely, and
    saying so is more useful to an engineer than another decimal of F1.
    """
    for ratio in sorted(ratios):
        if best_operating_point(y_true, scores, prevalence, ratio).worth_deploying:
            return float(ratio)
    return None
