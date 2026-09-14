"""One-class detection, threshold sensitivity, and transfer across task phases.

Three questions, three experiments
----------------------------------

**Can a detector that has never seen a failure find one?** Four lightweight
one-class models are fitted on the healthy training executions alone and scored
on the shared held-out set. They are compared on ROC-AUC and PR-AUC, which
depend on the ranking rather than on where a threshold happens to sit.

**What does not having labels cost?** A deployed detector cannot pick the
threshold that maximises F1, because computing that maximum needs the answers.
It can only place the threshold at a percentile of the healthy training scores.
Both are reported: the oracle maximum, and the F1 the percentile rule actually
achieves. The gap between them is the honest cost of the missing labels.

**Does it survive a change of task?** Leave-one-subset-out asks the harder
question: fitted on four phases of the assembly task, does the detector still
rank failures above healthy runs on a phase it has never seen? Nothing else in
this repository measures transfer, and transfer is what decides whether any of
this leaves the bench.

Holding out a subset is not enough on its own. LP2 and LP3 annotate the same 47
recordings, and LP4 and LP5 share 116 more, so removing LP3 from the training
data leaves every one of its traces behind under an LP2 label. The study is run
both ways and the gap between them is reported: naive subset hold-out, and
trace-disjoint hold-out where every copy of a held-out trace is removed too.

Every experiment here uses the grouped split as its headline, where no trace
appears on both sides of the partition. The random split is reported beside it
because it is what the published results on this dataset use.

Usage
-----

    make experiments
    python scripts/experiments.py --output reports/experiments.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config, set_seed, setup_logging
from src.data.loader import duplicate_summary, encode_labels, load_robot_data, trace_ids
from src.evaluation.metrics import confusion, score, threshold_curve
from src.evaluation.protocol import grouped_split, random_split
from src.features.statistical import create_statistical_features, feature_matrix
from src.models.detectors import (
    IsolationForestDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
)

logger = logging.getLogger("experiments")

#: Percentiles swept in the threshold sensitivity study. The range starts at 80
#: because below it the detector alarms on a fifth of the healthy runs it was
#: fitted on, which no line would accept, and stops short of 100 because the
#: last percentile is a single training point.
PERCENTILE_SWEEP = (80, 85, 90, 92.5, 95, 97.5, 99)


def build_detectors(config: Config) -> list:
    """The four detectors, each built from the configuration file."""
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
            percentile=config.threshold_percentile,
            random_state=config.random_state,
        ),
    ]


def load_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """The matrix, the target, the subset of origin, the trace identity, and the duplicate count."""
    frame, _ = encode_labels(load_robot_data(), binary=True)
    features = create_statistical_features(frame)
    X, _ = feature_matrix(features)
    return (
        X,
        frame["label_encoded"].to_numpy(dtype=int),
        frame["source"].to_numpy(),
        trace_ids(frame),
        duplicate_summary(frame),
    )


def run_one_class(X_train, y_train, X_test, y_test, config: Config) -> dict:
    """Fit on healthy training runs only, then score the shared test set."""
    healthy = X_train[y_train == 0]
    results = {}

    for detector in build_detectors(config):
        detector.fit(healthy)
        test_scores = detector.anomaly_score(X_test)
        analysis = threshold_curve(y_test, test_scores, detector.train_scores_)
        predicted = (test_scores > detector.threshold_).astype(int)

        results[detector.name] = {
            **score(y_test, predicted, test_scores),
            **confusion(y_test, predicted),
            **analysis.to_dict(),
            "operating_threshold": detector.threshold_,
            "operating_percentile": detector.percentile,
            "n_train_healthy": len(healthy),
        }
        if isinstance(detector, PCAReconstructionDetector):
            results[detector.name]["n_components"] = detector.n_components

        logger.info(
            "%-20s roc_auc=%.4f pr_auc=%.4f f1@p%.0f=%.4f best_f1=%.4f",
            detector.name,
            analysis.roc_auc,
            analysis.pr_auc,
            detector.percentile,
            results[detector.name]["f1_anomaly"],
            analysis.best_f1,
        )

    return results


def run_threshold_sweep(X_train, y_train, X_test, y_test, config: Config) -> dict:
    """F1 as a function of where the alarm threshold is placed.

    The detector is fitted once per percentile because the threshold is part of
    the fit, not of the scoring. Refitting is cheap here and avoids a subtle
    inconsistency between the stored threshold and the one being evaluated.
    """
    healthy = X_train[y_train == 0]
    sweep: dict[str, dict[str, float]] = {}

    for detector in build_detectors(config):
        detector.fit(healthy)
        test_scores = detector.anomaly_score(X_test)
        curve = {}
        for percentile in PERCENTILE_SWEEP:
            cutoff = float(np.percentile(detector.train_scores_, percentile))
            predicted = (test_scores > cutoff).astype(int)
            measured = score(y_test, predicted)
            curve[str(percentile)] = {
                "f1_anomaly": measured["f1_anomaly"],
                "precision_anomaly": measured["precision_anomaly"],
                "recall_anomaly": measured["recall_anomaly"],
                "threshold": cutoff,
            }
        sweep[detector.name] = curve

    return sweep


def run_transfer(X, y, sources, groups, config: Config, trace_disjoint: bool = True) -> dict:
    """Leave-one-subset-out: fit on four task phases, test on the fifth.

    This measures whether the healthy region learned on four phases of the
    assembly task still contains the healthy runs of a phase never seen. It is
    the closest this dataset can come to asking whether the detector transfers.

    ``trace_disjoint`` decides whether removing a subset actually removes its
    recordings. LP2 and LP3 annotate the same executions, so a naive hold-out of
    LP3 leaves all 47 of its traces in the training data wearing LP2 labels, and
    the detector is then tested on runs it has already seen. With the flag set,
    every copy of a held-out trace is removed from the training half too.
    """
    results: dict[str, dict] = {}

    for held_out in sorted(set(sources)):
        test_mask = sources == held_out
        train_mask = ~test_mask
        if trace_disjoint:
            train_mask &= ~np.isin(groups, np.unique(groups[test_mask]))
        healthy = X[train_mask & (y == 0)]

        if len(healthy) < 20 or len(set(y[test_mask])) < 2:
            logger.warning("subset %s skipped: not enough healthy or no contrast", held_out)
            continue

        per_detector = {}
        for detector in build_detectors(config):
            detector.fit(healthy)
            test_scores = detector.anomaly_score(X[test_mask])
            analysis = threshold_curve(y[test_mask], test_scores, detector.train_scores_)
            predicted = (test_scores > detector.threshold_).astype(int)
            per_detector[detector.name] = {
                **score(y[test_mask], predicted, test_scores),
                **analysis.to_dict(),
            }

        results[held_out] = {
            "n_train_healthy": len(healthy),
            "trace_disjoint": trace_disjoint,
            "n_test": int(test_mask.sum()),
            "anomaly_share_test": float(y[test_mask].mean()),
            "detectors": per_detector,
        }
        best = max(per_detector.items(), key=lambda item: item[1]["roc_auc"])
        logger.info(
            "held out %s: %d executions, best %s roc_auc=%.4f",
            held_out,
            int(test_mask.sum()),
            best[0],
            best[1]["roc_auc"],
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=None, help="path of the JSON report")
    parser.add_argument("--config", type=Path, default=None, help="path of the TOML configuration")
    parser.add_argument("--skip-transfer", action="store_true", help="one-class study only")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config.random_state)

    X, y, sources, groups, duplicates = load_matrix()
    logger.info(
        "%d executions over %d distinct traces (%.0f%% duplicated)",
        duplicates["n_executions"],
        duplicates["n_distinct_traces"],
        duplicates["duplicate_share"] * 100,
    )

    grouped = grouped_split(X, y, groups, config.cv_folds, config.random_state)
    random = random_split(X, y, groups, config.test_size, config.random_state)

    logger.info(
        "one-class study, grouped split: %d healthy training runs, %d held out",
        int((grouped.y_train == 0).sum()),
        len(grouped.y_test),
    )
    one_class = run_one_class(
        grouped.X_train, grouped.y_train, grouped.X_test, grouped.y_test, config
    )

    logger.info("the same detectors under the random split, for comparison")
    one_class_random = run_one_class(
        random.X_train, random.y_train, random.X_test, random.y_test, config
    )

    logger.info("threshold sensitivity over percentiles %s", list(PERCENTILE_SWEEP))
    sweep = run_threshold_sweep(
        grouped.X_train, grouped.y_train, grouped.X_test, grouped.y_test, config
    )

    transfer: dict = {}
    transfer_naive: dict = {}
    if config.transfer_enabled and not args.skip_transfer:
        logger.info("leave-one-subset-out, trace-disjoint")
        transfer = run_transfer(X, y, sources, groups, config, trace_disjoint=True)
        logger.info("leave-one-subset-out, naive, copies of held-out traces left in")
        transfer_naive = run_transfer(X, y, sources, groups, config, trace_disjoint=False)

    report = {
        "protocol": {
            "headline_split": "grouped",
            "n_total": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train_healthy": int((grouped.y_train == 0).sum()),
            "n_test": len(grouped.y_test),
            "anomaly_share_test": float(grouped.y_test.mean()),
            "percentile_sweep": list(PERCENTILE_SWEEP),
        },
        "duplicates": duplicates,
        "splits": {"grouped": grouped.summary(), "random": random.summary()},
        "config": config.to_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "one_class": one_class,
        "one_class_random_split": one_class_random,
        "threshold_sweep": sweep,
        "transfer": transfer,
        "transfer_naive": transfer_naive,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        logger.info("report written to %s", args.output)

    best = max(one_class.items(), key=lambda item: item[1]["roc_auc"])
    logger.info("best one-class detector by ROC-AUC: %s (%.4f)", best[0], best[1]["roc_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
