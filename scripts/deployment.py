"""What the detector would cost to run, and what it would be worth.

Two questions a results table never answers
-------------------------------------------

**Is it worth deploying?** F1 weighs a missed failure and a false alarm equally,
and no plant does. This script treats the failure rate of the line and the cost
of a missed failure relative to a false alarm as free parameters, then finds the
threshold that minimises expected cost across that grid. It reports the answer
that matters to an engineer: below which cost ratio the detector is not worth
switching on.

The failure rate is imposed from outside on purpose. This dataset is 72%
failures and a real line is not; leaving that inversion in the limitations
section and moving on would waste the one analysis that corrects for it.

**What does it cost to run?** Every score in this repository is produced by a
model small enough to sit on a controller, and "lightweight" has been asserted
here more often than it has been measured. Inference time per execution and the
serialised size of each fitted detector are measured, with the caveat that a
laptop CPU is not an embedded target and only the ratios between detectors
should be read across hardware.

Scores are pooled out of fold. One grouped five-fold pass gives every execution
a score from a detector that never saw its trace, so the cost analysis runs on
all 463 rather than on one held-out fifth.

Usage
-----

    make deployment
    python scripts/deployment.py --output reports/deployment.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config, set_seed, setup_logging
from src.data.loader import encode_labels, load_robot_data, trace_ids
from src.evaluation.calibration import percentile_threshold, tolerance_threshold
from src.evaluation.cost import (
    COST_RATIOS,
    PREVALENCES,
    best_operating_point,
    break_even_cost_ratio,
    cost_grid,
)
from src.features.statistical import create_statistical_features, feature_matrix
from src.models.detectors import (
    IsolationForestDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
)

logger = logging.getLogger("deployment")

#: Repeats of the timing loop. Inference here is fast enough that a single call
#: measures the clock rather than the model.
TIMING_REPEATS = 200

#: The reference plant for the headline figures: one failure per hundred
#: executions, a missed failure worth a hundred unnecessary stops. Both are
#: assumptions, both are stated, and the full grid is published beside them.
REFERENCE_PREVALENCE = 0.01
REFERENCE_RATIO = 100


def build_detectors(config: Config) -> list:
    return [
        IsolationForestDetector(
            percentile=config.threshold_percentile,
            random_state=config.random_state,
            n_estimators=config.isolation_forest_trees,
        ),
        OneClassSVMDetector(
            percentile=config.threshold_percentile,
            random_state=config.random_state,
            nu=config.one_class_svm_nu,
        ),
        PCAReconstructionDetector(
            percentile=config.threshold_percentile,
            random_state=config.random_state,
            variance=config.pca_variance,
        ),
        MahalanobisDetector(
            percentile=config.threshold_percentile, random_state=config.random_state
        ),
    ]


def out_of_fold_scores(X, y, groups, config: Config) -> tuple[dict, dict]:
    """Give every execution a score, and a decision, from a detector blind to its trace.

    The decisions matter as much as the scores. A cost curve drawn over every
    possible threshold shows what the detector *could* be worth; a deployment
    only gets the threshold a label-free rule actually picks. Both are recorded
    here so the gap between them can be priced instead of assumed small.
    """
    names = [detector.name for detector in build_detectors(config)]
    scores = {name: np.full(len(y), np.nan) for name in names}
    decided = {
        name: {rule: np.full(len(y), -1, dtype=int) for rule in ("percentile", "tolerance")}
        for name in names
    }

    splitter = StratifiedGroupKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
    )
    target_far = 1 - config.threshold_percentile / 100

    for train_index, test_index in splitter.split(X, y, groups=groups):
        healthy = X[train_index][y[train_index] == 0]
        for detector in build_detectors(config):
            detector.fit(healthy)
            fold_scores = detector.anomaly_score(X[test_index])
            scores[detector.name][test_index] = fold_scores

            rules = {
                "percentile": percentile_threshold(detector.train_scores_, target_far),
                "tolerance": tolerance_threshold(
                    detector.train_scores_, target_far, confidence=0.90
                ),
            }
            for rule_name, rule in rules.items():
                decided[detector.name][rule_name][test_index] = (fold_scores > rule.value).astype(
                    int
                )

    for name, values in scores.items():
        if np.isnan(values).any():
            raise RuntimeError(f"{name} left {int(np.isnan(values).sum())} executions unscored")
    return scores, decided


def cost_at_decisions(y: np.ndarray, predicted: np.ndarray, prevalence: float, ratio: float):
    """Expected cost of the decisions a label-free rule actually produced.

    Measured rates, imposed economics, same as everywhere else: the true and
    false positive rates come from the held-out executions, the failure rate of
    the line does not.
    """
    tpr = float(predicted[y == 1].mean())
    fpr = float(predicted[y == 0].mean())
    return {
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "expected_cost": prevalence * (1 - tpr) * ratio + (1 - prevalence) * fpr,
    }


def measure_inference_cost(X, y, config: Config) -> dict:
    """Time per execution and serialised size, per detector.

    The measurement is a laptop CPU and says nothing directly about a
    microcontroller. What it does say is the ratio between the four, and that
    ratio is a property of the models rather than of the machine: a Mahalanobis
    score is one quadratic form, an Isolation Forest score walks 200 trees.
    """
    healthy = X[y == 0]
    one = X[:1]
    measured = {}

    for detector in build_detectors(config):
        fit_start = time.perf_counter()
        detector.fit(healthy)
        fit_seconds = time.perf_counter() - fit_start

        detector.anomaly_score(one)  # warm any lazy allocation before timing
        start = time.perf_counter()
        for _ in range(TIMING_REPEATS):
            detector.anomaly_score(one)
        per_call = (time.perf_counter() - start) / TIMING_REPEATS

        batch_start = time.perf_counter()
        detector.anomaly_score(X)
        batch_seconds = time.perf_counter() - batch_start

        measured[detector.name] = {
            "fit_seconds": fit_seconds,
            "single_execution_ms": per_call * 1000,
            "per_execution_in_batch_ms": batch_seconds / len(X) * 1000,
            "serialised_bytes": len(pickle.dumps(detector)),
            "n_train_healthy": len(healthy),
        }
        logger.info(
            "%-20s %7.3f ms per execution, %7.1f kB on disk, fitted in %.3f s",
            detector.name,
            measured[detector.name]["single_execution_ms"],
            measured[detector.name]["serialised_bytes"] / 1024,
            fit_seconds,
        )

    return measured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=None, help="path of the JSON report")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config.random_state)

    frame, _ = encode_labels(load_robot_data(), binary=True)
    X, _ = feature_matrix(create_statistical_features(frame))
    y = frame["label_encoded"].to_numpy(dtype=int)
    groups = trace_ids(frame)

    logger.info("pooling out-of-fold scores over %d grouped folds", config.cv_folds)
    scores, decided = out_of_fold_scores(X, y, groups, config)

    economics = {}
    for name, values in scores.items():
        grid = cost_grid(y, values)
        break_even = {
            str(prevalence): break_even_cost_ratio(y, values, prevalence)
            for prevalence in PREVALENCES
        }
        # The reference case: a line failing once in a hundred executions, where a
        # missed failure costs a hundred unnecessary stops. Both numbers are
        # assumptions and both are stated.
        reference = best_operating_point(y, values, REFERENCE_PREVALENCE, REFERENCE_RATIO)
        deployed = {
            rule: cost_at_decisions(y, predictions, REFERENCE_PREVALENCE, REFERENCE_RATIO)
            for rule, predictions in decided[name].items()
        }

        economics[name] = {
            "grid": grid,
            "break_even_cost_ratio": break_even,
            "reference_case": {
                "prevalence": REFERENCE_PREVALENCE,
                "cost_ratio": REFERENCE_RATIO,
                "best_reachable": reference.to_dict(),
                "at_label_free_threshold": deployed,
                "price_of_miscalibration": {
                    rule: measured["expected_cost"] - reference.expected_cost
                    for rule, measured in deployed.items()
                },
            },
        }

        logger.info(
            "%-20s best reachable cost %.4f, percentile rule %.4f, tolerance rule %.4f",
            name,
            reference.expected_cost,
            deployed["percentile"]["expected_cost"],
            deployed["tolerance"]["expected_cost"],
        )

    logger.info("inference cost, laptop CPU, ratios are what transfer")
    inference = measure_inference_cost(X, y, config)

    report = {
        "protocol": {
            "n_total": len(y),
            "scores": "pooled out of fold, grouped five-fold",
            "prevalences": list(PREVALENCES),
            "cost_ratios": list(COST_RATIOS),
            "dataset_failure_rate": float(y.mean()),
            "timing_repeats": TIMING_REPEATS,
        },
        "config": config.to_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "economics": economics,
        "inference_cost": inference,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        logger.info("report written to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
