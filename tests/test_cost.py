"""The cost model, checked against the two policies it has to beat.

A detector is worth deploying only when its best operating point costs less than
both doing nothing and stopping on everything. These tests fix that definition
so it cannot drift into something more flattering.
"""

from __future__ import annotations

import numpy as np
from src.evaluation.cost import best_operating_point, break_even_cost_ratio, cost_grid


def separable():
    """A detector that ranks perfectly, and its labels."""
    y = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([np.linspace(0, 1, 100), np.linspace(2, 3, 100)])
    return y, scores


def useless():
    """A detector whose score carries no information about the label."""
    generator = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    return y, generator.normal(size=200)


def test_a_perfect_detector_beats_both_trivial_policies():
    y, scores = separable()
    point = best_operating_point(y, scores, prevalence=0.01, cost_ratio=100)
    assert point.worth_deploying
    assert point.expected_cost < point.cost_of_doing_nothing
    assert point.expected_cost < point.cost_of_always_stopping


def test_a_useless_detector_never_beats_them_by_much():
    """With no signal, the optimum collapses onto a trivial policy.

    The assertion is on the margin rather than on the verdict: a random score can
    look marginally better than a trivial policy by chance on 200 samples, and a
    test that forbade that would be testing the random seed.
    """
    y, scores = useless()
    point = best_operating_point(y, scores, prevalence=0.01, cost_ratio=100)
    trivial = min(point.cost_of_doing_nothing, point.cost_of_always_stopping)
    assert point.expected_cost > 0.5 * trivial


def test_a_dearer_missed_failure_never_makes_the_detector_less_useful():
    """Monotonicity, which is the sanity check the cost model must pass."""
    y, scores = separable()
    savings = [
        best_operating_point(y, scores, 0.01, ratio).to_dict()["saving_against_best_trivial"]
        for ratio in (1, 10, 100)
    ]
    assert savings == sorted(savings)


def test_the_break_even_ratio_is_the_first_one_that_pays():
    y, scores = separable()
    ratio = break_even_cost_ratio(y, scores, prevalence=0.01)
    assert ratio is not None
    assert best_operating_point(y, scores, 0.01, ratio).worth_deploying


def test_the_grid_covers_every_assumption_pair():
    y, scores = separable()
    grid = cost_grid(y, scores, prevalences=(0.01, 0.05), cost_ratios=(1, 10, 100))
    assert len(grid) == 6
    assert {row["prevalence"] for row in grid} == {0.01, 0.05}
