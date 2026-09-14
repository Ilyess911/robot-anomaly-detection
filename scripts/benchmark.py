"""Supervised benchmark: real baselines, a clean protocol, and the cost of a leak.

What this script establishes
----------------------------

The original notebooks published scores with nothing to read them against. This
script measures what a non-model gets, then what a stupid model gets, and only
then what the tuned models get. The distance between those three levels is the
only result that means anything.

It also fixes four defects of the original chain:

1. **A labelling error.** LP3 names its healthy class ``ok`` and never uses
   ``normal``. The binary encoder matched on ``normal`` alone, so the 20 healthy
   LP3 executions counted as failures and that subset looked 100% faulty. Fixed
   in ``src/data/loader.py``: 129 healthy executions out of 463, not 109.

2. **No baseline.** The constant answer, the best single feature at a single
   threshold, and a depth-2 tree are all reported first.

3. **An F1 whose definition moved.** Notebook 03 printed a weighted average,
   notebook 05 the positive-class F1, and notebook 04 both for the same model.
   Here ``f1_anomaly`` is the reference and ``f1_weighted`` is named.

4. **Two different test sets.** The supervised models were scored on 93
   executions and the unsupervised ones on 376. Every model in this project now
   shares one held-out set.

The duplicate leak
------------------

The five subsets are not five recordings. LP2 and LP3 annotate the same 47
executions under two fault taxonomies, LP4 and LP5 share 116 more, and the
merged frame holds 463 rows over 251 distinct traces. Under a random split, 69%
of the test set has an exact copy in the training half, which is where the
perfect scores came from.

Every model is therefore evaluated twice: once under the random split that
every published result on this dataset uses, and once under a grouped split
where no trace appears on both sides. Both splits hold 370 training and 93 test
executions at the same class balance, so the only variable between them is the
grouping.

The standardisation leak
------------------------

The notebooks standardise before they split, so the scaler sees the test half.
This script runs both protocols in the *same* environment with the same seeds,
which is the only way to attribute a gap to the leak rather than to five years
of scikit-learn releases.

One-class detection has moved to ``scripts/experiments.py``, which scores it on
continuous anomaly scores rather than on a fixed decision, and which runs the
threshold and transfer studies.

Usage
-----

    make benchmark
    python scripts/benchmark.py --output reports/benchmark.json --config configs/default.toml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config, set_seed, setup_logging
from src.data.loader import duplicate_summary, encode_labels, load_robot_data, trace_ids
from src.evaluation.metrics import confusion, score
from src.evaluation.protocol import grouped_split, random_split
from src.features.statistical import create_statistical_features, feature_matrix
from src.models.supervised import build_grids, decision_stump, search, shallow_tree

logger = logging.getLogger("benchmark")

# Kept at module level: tests/test_protocol.py imports them, and the published
# numbers are only reproducible if these are the values that produced them.
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

#: Names of the 48 features, in column order. Filled by build_dataset so that
#: the single-feature baseline can say which sensor statistic it used.
FEATURE_NAMES: list[str] = []


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load, label and featurise, returning the model matrix and the target."""
    X, y, _ = build_dataset_with_groups()
    return X, y


def build_dataset_with_groups() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The matrix, the target, and the trace identity of every execution.

    The third array is what makes a grouped split possible: two rows share an
    identifier exactly when they are the same physical recording annotated
    twice.
    """
    frame, _ = encode_labels(load_robot_data(), binary=True)
    features = create_statistical_features(frame)
    X, columns = feature_matrix(features)

    FEATURE_NAMES.clear()
    FEATURE_NAMES.extend(columns)
    return (
        X,
        frame["label_encoded"].to_numpy(dtype=int),
        trace_ids(frame),
    )


def run_baselines(y_test: np.ndarray, X_train=None, y_train=None, X_test=None) -> dict:
    """The baselines without which no score can be read.

    Three levels of decreasing stupidity, and it is the distance between them
    that says what a model actually contributes:

    1. always answer "anomaly". The set is 72% failures, so this non-model
       already scores highly on the positive class;
    2. one sensor, one threshold. This is the baseline almost nobody publishes,
       and it is the most instructive: if it suffices, the problem was not hard;
    3. a depth-2 tree, three decisions in total.

    A grid-searched model is not judged against zero. It is judged against
    these.
    """
    results = {
        "always_anomaly": score(y_test, np.ones_like(y_test)),
        "always_normal": score(y_test, np.zeros_like(y_test)),
    }

    if X_train is None:
        return results

    best = None
    for index in range(X_train.shape[1]):
        stump = decision_stump(RANDOM_STATE).fit(X_train[:, [index]], y_train)
        current = score(y_test, stump.predict(X_test[:, [index]]))
        if best is None or current["f1_anomaly"] > best[0]["f1_anomaly"]:
            best = (current, index)

    name = FEATURE_NAMES[best[1]] if best[1] < len(FEATURE_NAMES) else str(best[1])
    results["best_single_feature"] = {**best[0], "feature": name}

    tree = shallow_tree(RANDOM_STATE).fit(X_train, y_train)
    results["depth_2_tree"] = score(y_test, tree.predict(X_test))

    return results


def run_supervised(
    X_train,
    X_test,
    y_train,
    y_test,
    config: Config,
    leaky: bool = False,
    groups_train=None,
) -> dict:
    """Search then evaluate each classifier.

    ``leaky=False`` puts the scaler inside the Pipeline, so it is refitted on
    the training folds alone at every turn of the cross-validation.

    ``leaky=True`` reproduces the notebooks' defect: the data arrives already
    standardised over the whole dataset, the scaler having seen the test half.

    ``groups_train`` switches the inner cross-validation to a grouped one, so
    that a search run under the grouped protocol does not put the duplicates
    back inside the folds it selects on.

    Both modes run in the same environment with the same seeds and the same
    grids. Comparing a notebook executed in 2025 under Python 3.9 against a
    script run today would mix two variables and prove nothing.
    """
    results = {}

    for name, (estimator, grid) in build_grids(config.random_state).items():
        fitted = search(
            estimator,
            grid,
            X_train,
            y_train,
            folds=config.cv_folds,
            random_state=config.random_state,
            scale=not leaky,
            groups=groups_train,
        )

        y_pred = fitted.predict(X_test)
        y_proba = fitted.predict_proba(X_test)[:, 1]

        results[name] = {
            **score(y_test, y_pred, y_proba),
            **confusion(y_test, y_pred),
            "cv_f1_mean": fitted.best_score_,
            "cv_f1_std": fitted.cv_results_["std_test_score"][fitted.best_index_],
            "best_params": {k.replace("model__", ""): v for k, v in fitted.best_params_.items()},
        }
        logger.info("%-20s f1(anomaly)=%.4f", name, results[name]["f1_anomaly"])

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=None, help="path of the JSON report")
    parser.add_argument("--config", type=Path, default=None, help="path of the TOML configuration")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config.random_state)

    X, y, groups = build_dataset_with_groups()
    frame_duplicates = duplicate_summary(encode_labels(load_robot_data(), binary=True)[0])
    logger.info(
        "%d executions over %d distinct traces (%.0f%% duplicated)",
        frame_duplicates["n_executions"],
        frame_duplicates["n_distinct_traces"],
        frame_duplicates["duplicate_share"] * 100,
    )

    random_protocol = random_split(X, y, groups, config.test_size, config.random_state)
    grouped_protocol = grouped_split(X, y, groups, config.cv_folds, config.random_state)

    logger.info("baselines on the grouped test set, from the stupidest up")
    baselines = run_baselines(
        grouped_protocol.y_test,
        grouped_protocol.X_train,
        grouped_protocol.y_train,
        grouped_protocol.X_test,
    )
    for name, value in baselines.items():
        logger.info(
            "%-22s f1(anomaly)=%.4f accuracy=%.4f",
            name,
            value["f1_anomaly"],
            value["accuracy"],
        )

    baselines_random = run_baselines(
        random_protocol.y_test,
        random_protocol.X_train,
        random_protocol.y_train,
        random_protocol.X_test,
    )

    # The notebooks' second defect: standardise BEFORE splitting, so the scaler
    # sees the test half. Measured on the random protocol, which is the one the
    # notebooks used, and in this same environment so that the gap is the leak
    # and not five years of scikit-learn releases.
    X_leaky = StandardScaler().fit_transform(X)
    leaky_protocol = random_split(X_leaky, y, groups, config.test_size, config.random_state)

    logger.info("supervised, random split, scaler fitted on the whole dataset")
    leaky = run_supervised(
        leaky_protocol.X_train,
        leaky_protocol.X_test,
        leaky_protocol.y_train,
        leaky_protocol.y_test,
        config,
        leaky=True,
    )

    logger.info("supervised, random split, scaler inside the pipeline")
    supervised_random = run_supervised(
        random_protocol.X_train,
        random_protocol.X_test,
        random_protocol.y_train,
        random_protocol.y_test,
        config,
    )

    logger.info("supervised, grouped split, no trace on both sides")
    supervised = run_supervised(
        grouped_protocol.X_train,
        grouped_protocol.X_test,
        grouped_protocol.y_train,
        grouped_protocol.y_test,
        config,
        groups_train=grouped_protocol.groups_train,
    )

    logger.info("what the standardisation leak was worth, at constant environment")
    for name in supervised_random:
        before = leaky[name]["f1_anomaly"]
        after = supervised_random[name]["f1_anomaly"]
        logger.info("%-22s leaky %.4f  clean %.4f  %+.4f", name, before, after, after - before)

    logger.info("what the duplicate leak was worth, same features, same grids")
    for name in supervised:
        before = supervised_random[name]["f1_anomaly"]
        after = supervised[name]["f1_anomaly"]
        logger.info("%-22s random %.4f  grouped %.4f  %+.4f", name, before, after, after - before)

    report = {
        "protocol": {
            "n_total": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": len(grouped_protocol.y_train),
            "n_test": len(grouped_protocol.y_test),
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "random_state": config.random_state,
            "scaler_inside_cv": True,
            "headline_split": "grouped",
            "anomaly_share_test": float(grouped_protocol.y_test.mean()),
        },
        "duplicates": frame_duplicates,
        "splits": {
            "random": random_protocol.summary(),
            "grouped": grouped_protocol.summary(),
        },
        "config": config.to_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "baselines": baselines,
        "baselines_random": baselines_random,
        "supervised": supervised,
        "supervised_random": supervised_random,
        "supervised_leaky": leaky,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        logger.info("report written to %s", args.output)

    best = max(supervised.items(), key=lambda item: item[1]["f1_anomaly"])
    logger.info(
        "best supervised model, grouped split: %s (f1 %.4f)", best[0], best[1]["f1_anomaly"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
