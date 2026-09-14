"""Does the threshold rule keep the promise it makes.

One rule here promises nothing and one promises a bound. The tests check
exactly that asymmetry, because the whole calibration argument of this project
rests on it: the tolerance bound is not merely more conservative in practice, it
is conservative by construction, and that is provable rather than observed.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.evaluation.calibration import (
    calibration_gap,
    percentile_threshold,
    realised_far,
    tolerance_threshold,
)

TARGET = 0.05
CONFIDENCE = 0.90
N_TRAIN = 103  # the healthy training runs this dataset actually offers


@pytest.mark.parametrize("distribution", ["normal", "exponential", "heavy_tailed"])
def test_the_tolerance_bound_holds_whatever_the_score_distribution(distribution):
    """Distribution-free means distribution-free, so three shapes are tried.

    The guarantee is that the realised false alarm rate exceeds the target in at
    most 10% of repetitions. The check is run over 200 draws and allows a margin
    for the randomness of the check itself, which is the honest way to test a
    probabilistic claim.
    """
    generator = np.random.default_rng(0)
    draw = {
        "normal": lambda n: generator.normal(size=n),
        "exponential": lambda n: generator.exponential(size=n),
        "heavy_tailed": lambda n: generator.standard_t(df=2, size=n),
    }[distribution]

    exceeded = 0
    repetitions = 200
    for _ in range(repetitions):
        train = draw(N_TRAIN)
        held_out = draw(2000)
        threshold = tolerance_threshold(train, TARGET, CONFIDENCE)
        if realised_far(threshold, held_out) > TARGET:
            exceeded += 1

    rate = exceeded / repetitions
    assert rate <= 1 - CONFIDENCE + 0.05, (
        f"{distribution}: the bound was broken in {rate:.0%} of draws, "
        f"more than the {1 - CONFIDENCE:.0%} it allows"
    )


def test_the_percentile_rule_breaks_its_target_far_more_often():
    """The baseline promises nothing, and this measures how little.

    This is not a failing test dressed as a passing one. The percentile of a
    small sample is an estimate of a tail quantile, and half the time an
    estimate lands on the low side. Showing that it misses roughly half the time
    is what justifies preferring a bound.
    """
    generator = np.random.default_rng(1)
    exceeded = 0
    repetitions = 200
    for _ in range(repetitions):
        train = generator.normal(size=N_TRAIN)
        held_out = generator.normal(size=2000)
        threshold = percentile_threshold(train, TARGET)
        if realised_far(threshold, held_out) > TARGET:
            exceeded += 1

    assert exceeded / repetitions > 0.3, "the percentile rule is expected to miss often"


def test_the_bound_costs_sensitivity():
    """A guarantee is paid for with a higher threshold, never with nothing."""
    generator = np.random.default_rng(2)
    train = generator.normal(size=N_TRAIN)
    assert (
        tolerance_threshold(train, TARGET, CONFIDENCE).value
        >= percentile_threshold(train, TARGET).value
    )


def test_a_sample_too_small_for_the_guarantee_says_so():
    """Twenty samples cannot support a 5% bound at 90% confidence.

    The rule returns the largest observation and reports the bound it actually
    achieves, so a caller can see the guarantee was not met. Returning a
    threshold while silently claiming the requested bound would be the dangerous
    behaviour.
    """
    generator = np.random.default_rng(3)
    threshold = tolerance_threshold(generator.normal(size=20), TARGET, CONFIDENCE)
    assert threshold.order_statistic == 1
    assert threshold.guaranteed_far > TARGET


def test_the_gap_report_names_a_target_that_was_missed():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    threshold = percentile_threshold(np.zeros(10), TARGET)
    gap = calibration_gap(threshold, scores)
    assert gap["realised_far"] > TARGET
    assert gap["exceeds_target"] is True
