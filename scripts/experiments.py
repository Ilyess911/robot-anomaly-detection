"""One-class detection, threshold sensitivity, and transfer across task phases.

Three questions, three experiments
----------------------------------

**Can a detector that has never seen a failure find one?** Four lightweight
one-class models are fitted on healthy executions alone and scored under
repeated grouped cross-validation, five repeats of five folds, so every figure
comes with a confidence interval over 25 fits rather than from one lucky split.
Two detectors whose intervals overlap are reported as indistinguishable instead
of ranked.

**What does not having labels cost, and can the threshold be made to promise
something?** A deployed detector cannot pick the threshold that maximises F1,
since computing that maximum needs the answers. Two label-free rules are
compared on every fold: the usual percentile of the healthy training scores,
which guarantees nothing, and a distribution-free tolerance bound that caps the
false alarm rate with stated confidence. What each one promises and what each
one delivers on held-out healthy runs are reported side by side.

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
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config, set_seed, setup_logging
from src.data.loader import duplicate_summary, encode_labels, load_robot_data, trace_ids
from src.evaluation.calibration import calibration_gap, percentile_threshold, tolerance_threshold
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

#: Repeats of the grouped five-fold cross-validation. Five repeats give 25 fits
#: per detector, enough for a confidence interval that is not itself noise, and
#: the detectors are cheap enough that the whole study runs in seconds.
REPEATS = 5


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


def run_repeated_cv(X, y, groups, config: Config, repeats: int = 5, grouped: bool = True) -> dict:
    """The headline one-class numbers: repeated grouped cross-validation.

    A single held-out fold of 93 executions cannot separate 0.918 from 0.899,
    and reporting four detectors at ROC-AUC 1.000 on one split invites exactly
    the scepticism it deserves. This runs five repeats of grouped five-fold
    cross-validation, so every detector is fitted and scored 25 times on
    partitions where no sensor trace appears on both sides.

    Each fold also carries the calibration study, because the threshold rules
    have to be judged on the same partitions as the scores they threshold. For
    every fold the two rules are calibrated on the healthy *training* runs and
    their realised false alarm rate is measured on the healthy *held-out* runs,
    which is the only measurement that says whether a rule works.

    Returns per-detector aggregates with a t-based 95% confidence interval on
    the mean over the 25 folds, so that two detectors whose intervals overlap
    can be called indistinguishable instead of ranked.
    """
    per_detector: dict[str, dict[str, list[float]]] = {}

    for repeat in range(repeats):
        seed = config.random_state + repeat
        if grouped:
            splitter = StratifiedGroupKFold(
                n_splits=config.cv_folds, shuffle=True, random_state=seed
            )
            folds = splitter.split(X, y, groups=groups)
        else:
            splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
            folds = splitter.split(X, y)

        for train_index, test_index in folds:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            healthy_train = X_train[y_train == 0]
            if len(healthy_train) < 30 or len(set(y_test)) < 2:
                continue

            for detector in build_detectors(config):
                detector.fit(healthy_train)
                test_scores = detector.anomaly_score(X_test)
                healthy_test = test_scores[y_test == 0]

                target = 1 - config.threshold_percentile / 100
                rules = {
                    "percentile": percentile_threshold(detector.train_scores_, target),
                    "tolerance": tolerance_threshold(
                        detector.train_scores_, target, confidence=0.90
                    ),
                }

                bucket = per_detector.setdefault(detector.name, {})
                analysis = threshold_curve(y_test, test_scores, detector.train_scores_)
                bucket.setdefault("roc_auc", []).append(analysis.roc_auc)
                bucket.setdefault("pr_auc", []).append(analysis.pr_auc)
                bucket.setdefault("best_f1", []).append(analysis.best_f1)

                for name, rule in rules.items():
                    predicted = (test_scores > rule.value).astype(int)
                    measured = score(y_test, predicted)
                    bucket.setdefault(f"f1_{name}", []).append(measured["f1_anomaly"])
                    bucket.setdefault(f"recall_{name}", []).append(measured["recall_anomaly"])
                    bucket.setdefault(f"far_{name}", []).append(
                        calibration_gap(rule, healthy_test)["realised_far"]
                    )

    results = {}
    for name, metrics in per_detector.items():
        results[name] = {metric: summarise(values) for metric, values in sorted(metrics.items())}
        results[name]["n_folds"] = len(metrics["roc_auc"])
        logger.info(
            "%-20s roc_auc %s  f1(percentile) %s  f1(tolerance) %s",
            name,
            format_interval(results[name]["roc_auc"]),
            format_interval(results[name]["f1_percentile"]),
            format_interval(results[name]["f1_tolerance"]),
        )

    return results


def summarise(values: list[float]) -> dict:
    """Mean, spread, and a t-based 95% confidence interval on the mean."""
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    n = len(array)
    mean = float(array.mean()) if n else float("nan")
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    half = float(scipy_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": float(array.min()) if n else float("nan"),
        "max": float(array.max()) if n else float("nan"),
        "n": n,
    }


def format_interval(summary: dict) -> str:
    return f"{summary['mean']:.3f} [{summary['ci95_low']:.3f}, {summary['ci95_high']:.3f}]"


def paired_comparison(cv: dict, metric: str = "f1_percentile") -> dict:
    """Are the detectors actually different, or is the ranking noise.

    Reports, for every pair, whether the 95% interval of one contains the mean
    of the other. Two detectors that fail that test are reported as
    indistinguishable on this data, which is the honest reading of a 0.02 gap
    over 93 executions.
    """
    names = sorted(cv)
    verdicts = {}
    for first in names:
        for second in names:
            if first >= second:
                continue
            a, b = cv[first][metric], cv[second][metric]
            overlap = a["ci95_low"] <= b["ci95_high"] and b["ci95_low"] <= a["ci95_high"]
            verdicts[f"{first} vs {second}"] = {
                "difference": a["mean"] - b["mean"],
                "intervals_overlap": bool(overlap),
                "distinguishable": not overlap,
            }
    return verdicts


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
        "repeated grouped cross-validation, %d repeats x %d folds",
        REPEATS,
        config.cv_folds,
    )
    cross_validated = run_repeated_cv(X, y, groups, config, repeats=REPEATS, grouped=True)
    comparison = paired_comparison(cross_validated)

    logger.info("the same study without grouping, to price the duplicates again")
    cross_validated_ungrouped = run_repeated_cv(
        X, y, groups, config, repeats=REPEATS, grouped=False
    )
    for name, grouped_metrics in cross_validated.items():
        ungrouped = cross_validated_ungrouped[name]["far_percentile"]["mean"]
        logger.info(
            "%-20s realised false alarm rate: grouped %.1f%%, ungrouped %.1f%%, target %.1f%%",
            name,
            grouped_metrics["far_percentile"]["mean"] * 100,
            ungrouped * 100,
            (100 - config.threshold_percentile),
        )
    for pair, verdict in comparison.items():
        logger.info(
            "%-46s %+.3f  %s",
            pair,
            verdict["difference"],
            "distinguishable" if verdict["distinguishable"] else "indistinguishable",
        )

    logger.info(
        "single illustrative fold: %d healthy training runs, %d held out",
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
        "cross_validated": cross_validated,
        "cross_validated_ungrouped": cross_validated_ungrouped,
        "detector_comparison": comparison,
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

    best = max(cross_validated.items(), key=lambda item: item[1]["f1_percentile"]["mean"])
    logger.info(
        "best one-class detector over %d folds: %s, f1 %s",
        best[1]["n_folds"],
        best[0],
        format_interval(best[1]["f1_percentile"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
