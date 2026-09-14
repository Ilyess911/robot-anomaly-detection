"""Score executions with a fitted detector. The inference side, kept separate.

Training, evaluation and inference are three different things and this file is
only the third. It fits a one-class detector on healthy executions, scores a set
of executions it has not seen, and prints them ranked by anomaly score with the
alarm threshold marked.

Two input modes:

``--subset LP1``
    Score one of the five subsets shipped with the repository. Training then
    excludes not just that subset but every copy of its traces, since LP2 and
    LP3 hold the same recordings and LP4 and LP5 share 116 more. Without that,
    the detector would be scoring runs it was fitted on.

``--file path/to/file.data``
    Score any file in the UCI layout: a label line, then fifteen lines of six
    readings. The label is used only to report whether the detector was right,
    never to fit anything.

Usage
-----

    python scripts/detect.py --subset LP1 --detector mahalanobis --top 15
    python scripts/detect.py --file data/lp4.data --output /tmp/scored.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, set_seed, setup_logging
from src.data.loader import (
    SUBSET_FILES,
    _parse_subset,
    encode_labels,
    load_robot_data,
    trace_ids,
)
from src.evaluation.metrics import score as score_metrics
from src.features.statistical import create_statistical_features, feature_matrix
from src.models.detectors import DETECTORS

logger = logging.getLogger("detect")


def build_detector(name: str, config, healthy: np.ndarray):
    """One detector from the registry, configured from the TOML file."""
    keyword = {
        "isolation_forest": {"n_estimators": config.isolation_forest_trees},
        "one_class_svm": {"nu": config.one_class_svm_nu},
        "pca_reconstruction": {"variance": config.pca_variance},
        "mahalanobis": {},
    }[name]
    detector = DETECTORS[name](
        percentile=config.threshold_percentile, random_state=config.random_state, **keyword
    )
    return detector.fit(healthy)


def read_single_file(path: Path) -> pd.DataFrame:
    """Parse one file in the UCI layout into the usual frame."""
    executions = _parse_subset(str(path), path.stem.upper())
    if not executions:
        raise SystemExit(f"no execution parsed from {path}")
    frame = pd.DataFrame(executions)
    columns = sorted(
        (c for c in frame.columns if c.startswith("feature_")),
        key=lambda name: int(name.split("_")[1]),
    )
    return frame[[*columns, "label", "source"]].copy()


def featurise(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Statistical features and the binary target for a loaded frame."""
    labelled, _ = encode_labels(frame, binary=True)
    X, _ = feature_matrix(create_statistical_features(labelled))
    return X, labelled["label_encoded"].to_numpy(dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--subset", choices=sorted(SUBSET_FILES), help="score one shipped subset")
    source.add_argument("--file", type=Path, help="score a file in the UCI layout")
    parser.add_argument("--detector", choices=sorted(DETECTORS), default="mahalanobis")
    parser.add_argument("--top", type=int, default=15, help="rows to print, worst first")
    parser.add_argument("--output", type=Path, default=None, help="write every score to a CSV")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config.random_state)

    reference, _ = encode_labels(load_robot_data(), binary=True)
    reference_X, reference_y = featurise(load_robot_data())
    reference_groups = trace_ids(reference)

    if args.file:
        target = read_single_file(args.file)
        target_X, target_y = featurise(target)
        # A file supplied from outside gets the whole healthy set to learn from.
        # If it happens to be one of the shipped subsets, that is the caller's
        # call to make, and the printed metrics say so.
        train_mask = np.ones(len(reference_y), dtype=bool)
        origin = str(args.file)
    else:
        subset = args.subset or "LP1"
        target_mask = reference["source"].to_numpy() == subset
        target_X, target_y = reference_X[target_mask], reference_y[target_mask]
        # Exclude every copy of the target traces, not just the subset itself.
        train_mask = ~np.isin(reference_groups, np.unique(reference_groups[target_mask]))
        origin = subset

    healthy = reference_X[train_mask & (reference_y == 0)]
    if len(healthy) < 20:
        logger.error("only %d healthy executions left to fit on, refusing", len(healthy))
        return 1

    detector = build_detector(args.detector, config, healthy)
    scores = detector.anomaly_score(target_X)
    flagged = (scores > detector.threshold_).astype(int)

    logger.info(
        "%s: %d executions scored by %s, fitted on %d healthy runs from elsewhere",
        origin,
        len(scores),
        args.detector,
        len(healthy),
    )

    table = pd.DataFrame(
        {
            "execution": np.arange(len(scores)),
            "anomaly_score": scores,
            "flagged": flagged,
            "actual": np.where(target_y == 1, "failure", "healthy"),
        }
    ).sort_values("anomaly_score", ascending=False)

    pd.set_option("display.width", 100)
    print(table.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        f"\nalarm threshold (p{detector.percentile:.0f} of healthy training scores): "
        f"{detector.threshold_:.3f}"
    )
    print(f"flagged {int(flagged.sum())} of {len(flagged)} executions")

    if len(set(target_y)) > 1:
        measured = score_metrics(target_y, flagged, scores)
        print(
            f"against the recorded labels: f1 {measured['f1_anomaly']:.3f}, "
            f"precision {measured['precision_anomaly']:.3f}, "
            f"recall {measured['recall_anomaly']:.3f}, "
            f"roc-auc {measured['roc_auc']:.3f}"
        )

    if args.output:
        table.to_csv(args.output, index=False)
        logger.info("scores written to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
