"""Regenerate every figure the README publishes, from the reports on disk.

No figure in this repository is drawn by hand or kept from a previous run. The
first version of this project committed its plots before a labelling fix and
then kept showing the old story next to corrected tables, which is the failure
mode this script exists to prevent: one command redraws everything, and a
figure that contradicts ``reports/`` cannot survive it.

The expensive grid searches are read from ``reports/benchmark.json`` rather than
refitted. Anything that needs a curve rather than a number, a ROC or a score
distribution, is recomputed here, which takes seconds.

Usage
-----

    make figures
    python scripts/figures.py --reports reports --output assets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, set_seed, setup_logging
from src.data.loader import SENSOR_NAMES, as_time_series, encode_labels, load_robot_data, trace_ids
from src.evaluation.metrics import confusion
from src.evaluation.protocol import grouped_split
from src.features.statistical import create_statistical_features, feature_matrix
from src.models.detectors import (
    IsolationForestDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
)

logger = logging.getLogger("figures")

HEALTHY = "#2E6F9E"
ANOMALY = "#C0392B"
ACCENT = "#E08A1E"
GREY = "#8A8F98"
DARK = "#1B222B"

DETECTOR_ORDER = ["isolation_forest", "one_class_svm", "pca_reconstruction", "mahalanobis"]
PRETTY = {
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "pca_reconstruction": "PCA reconstruction",
    "mahalanobis": "Mahalanobis",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm_rbf": "SVM (RBF)",
    "gradient_boosting": "Gradient Boosting",
    "always_anomaly": 'Always answer "anomaly"',
    "best_single_feature": "One sensor, one threshold",
    "depth_2_tree": "Depth-2 tree",
}


def style() -> None:
    """One look for every figure, set once."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#C9CED6",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E7EAEE",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def fit_detectors(X_train, y_train, config):
    """The four detectors, fitted on the healthy training runs."""
    healthy = X_train[y_train == 0]
    detectors = [
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
    for detector in detectors:
        detector.fit(healthy)
    return detectors


def figure_detection(split, detectors, out: Path) -> None:
    """Anomaly score per held-out execution, with the alarm line drawn on it.

    The figure the README opens with. It has to answer one question without a
    caption: are the failures separable from the healthy runs, and where does
    the alarm actually fall.
    """
    detector = next(d for d in detectors if d.name == "mahalanobis")
    scores = detector.anomaly_score(split.X_test)
    order = np.argsort(scores)
    y = split.y_test[order]
    scores = scores[order]

    figure, axis = plt.subplots(figsize=(9, 4.2))
    index = np.arange(len(scores))
    axis.scatter(
        index[y == 0], scores[y == 0], s=34, color=HEALTHY, label="healthy execution", zorder=3
    )
    axis.scatter(
        index[y == 1], scores[y == 1], s=34, color=ANOMALY, label="failed execution", zorder=3
    )
    axis.axhline(
        detector.threshold_,
        color=DARK,
        linestyle="--",
        linewidth=1.2,
        label=f"alarm threshold (p{detector.percentile:.0f} of healthy training scores)",
    )
    axis.set_yscale("log")
    axis.set_xlabel("held-out executions, sorted by anomaly score")
    axis.set_ylabel("anomaly score (log scale)")
    axis.set_title(
        "Mahalanobis distance to the healthy operating region, 93 unseen executions",
    )
    axis.legend(loc="upper left")
    figure.savefig(out / "detection-overview.png")
    plt.close(figure)


def figure_signals(frame, out: Path) -> None:
    """Mean sensor trace, healthy against failed, six channels."""
    traces = as_time_series(frame)
    healthy = traces[frame["label_binary"].to_numpy() == 0]
    failed = traces[frame["label_binary"].to_numpy() == 1]

    figure, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    steps = np.arange(traces.shape[1])
    for index, (axis, sensor) in enumerate(zip(axes.ravel(), SENSOR_NAMES, strict=True)):
        healthy_mean = healthy[:, :, index].mean(axis=0)
        failed_mean = failed[:, :, index].mean(axis=0)
        axis.plot(steps, healthy_mean, color=HEALTHY, linewidth=2, label="healthy")
        axis.plot(steps, failed_mean, color=ANOMALY, linewidth=2, label="failed")
        axis.fill_between(
            steps,
            healthy[:, :, index].mean(axis=0) - healthy[:, :, index].std(axis=0),
            healthy[:, :, index].mean(axis=0) + healthy[:, :, index].std(axis=0),
            color=HEALTHY,
            alpha=0.13,
        )
        axis.fill_between(
            steps,
            failed[:, :, index].mean(axis=0) - failed[:, :, index].std(axis=0),
            failed[:, :, index].mean(axis=0) + failed[:, :, index].std(axis=0),
            color=ANOMALY,
            alpha=0.13,
        )
        axis.set_title(sensor, fontsize=10)
        if index >= 3:
            axis.set_xlabel("time step")
    axes[0][0].set_ylabel("force")
    axes[1][0].set_ylabel("torque")
    handles = [
        Patch(color=HEALTHY, label=f"healthy ({len(healthy)} executions)"),
        Patch(color=ANOMALY, label=f"failed ({len(failed)} executions)"),
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    figure.suptitle("Mean force and torque per channel, shaded to one standard deviation", y=1.0)
    figure.tight_layout()
    figure.savefig(out / "healthy-vs-failed-signals.png")
    plt.close(figure)


def figure_duplicate_leak(benchmark: dict, out: Path) -> None:
    """Cross-validated F1 under the random split against the grouped one.

    The central result. Same features, same grids, same seeds; the only
    difference is whether copies of a trace are allowed on both sides.
    """
    names = [name for name in benchmark["supervised"] if name in benchmark["supervised_random"]]
    grouped = [benchmark["supervised"][name]["cv_f1_mean"] for name in names]
    grouped_err = [benchmark["supervised"][name]["cv_f1_std"] for name in names]
    random = [benchmark["supervised_random"][name]["cv_f1_mean"] for name in names]
    random_err = [benchmark["supervised_random"][name]["cv_f1_std"] for name in names]

    position = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.6, 4.2))
    axis.bar(
        position - width / 2,
        random,
        width,
        yerr=random_err,
        capsize=3,
        color=GREY,
        label="random split, copies on both sides",
    )
    axis.bar(
        position + width / 2,
        grouped,
        width,
        yerr=grouped_err,
        capsize=3,
        color=HEALTHY,
        label="grouped split, every trace on one side",
    )
    for x, (before, after) in enumerate(zip(random, grouped, strict=True)):
        axis.text(
            x,
            max(before, after) + 0.022,
            f"{after - before:+.3f}",
            ha="center",
            fontsize=9,
            color=ANOMALY if after < before else DARK,
        )
    axis.set_xticks(position)
    axis.set_xticklabels([PRETTY.get(name, name) for name in names], fontsize=9)
    axis.set_ylim(0.85, 1.045)
    axis.set_ylabel("cross-validated F1, anomaly class")
    axis.set_title("What 212 duplicated executions were worth")
    axis.legend(loc="lower left")
    figure.savefig(out / "duplicate-leak.png")
    plt.close(figure)


def figure_comparison(benchmark: dict, experiments: dict, out: Path) -> None:
    """Every approach on one axis, baselines included."""
    rows = [
        (PRETTY["always_anomaly"], benchmark["baselines"]["always_anomaly"]["f1_anomaly"], GREY),
        (
            PRETTY["best_single_feature"],
            benchmark["baselines"]["best_single_feature"]["f1_anomaly"],
            GREY,
        ),
        (PRETTY["depth_2_tree"], benchmark["baselines"]["depth_2_tree"]["f1_anomaly"], GREY),
    ]
    for name in DETECTOR_ORDER:
        if name in experiments["one_class"]:
            rows.append((PRETTY[name], experiments["one_class"][name]["f1_anomaly"], ACCENT))
    for name, value in benchmark["supervised"].items():
        rows.append((PRETTY.get(name, name), value["f1_anomaly"], HEALTHY))

    labels = [row[0] for row in rows][::-1]
    values = [row[1] for row in rows][::-1]
    colors = [row[2] for row in rows][::-1]

    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.barh(np.arange(len(values)), values, color=colors, height=0.66)
    for index, value in enumerate(values):
        axis.text(value + 0.006, index, f"{value:.3f}", va="center", fontsize=9)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_xlim(0, 1.09)
    axis.set_xlabel("F1 on the anomaly class, grouped held-out set of 93 executions")
    axis.set_title("Baselines, one-class detectors, supervised classifiers")
    axis.grid(axis="y", visible=False)
    handles = [
        Patch(color=GREY, label="baseline, not a model"),
        Patch(color=ACCENT, label="one-class, healthy runs only"),
        Patch(color=HEALTHY, label="supervised, labelled failures"),
    ]
    axis.legend(handles=handles, loc="lower right")
    figure.savefig(out / "model-comparison.png")
    plt.close(figure)


def figure_threshold(experiments: dict, out: Path) -> None:
    """F1 as the alarm threshold slides across the healthy score distribution."""
    sweep = experiments["threshold_sweep"]
    percentiles = [float(key) for key in next(iter(sweep.values()))]

    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    palette = {
        "isolation_forest": ANOMALY,
        "one_class_svm": HEALTHY,
        "pca_reconstruction": ACCENT,
        "mahalanobis": DARK,
    }
    for name in DETECTOR_ORDER:
        if name not in sweep:
            continue
        values = [sweep[name][key]["f1_anomaly"] for key in sweep[name]]
        axis.plot(
            percentiles,
            values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=palette[name],
            label=PRETTY[name],
        )
    axis.set_xlabel("threshold, as a percentile of the healthy training scores")
    axis.set_ylabel("F1, anomaly class")
    axis.set_title("Every detector ranks perfectly. Only the threshold separates them")
    axis.legend(loc="lower left", ncol=2)
    figure.savefig(out / "threshold-sensitivity.png")
    plt.close(figure)


def figure_curves(split, detectors, out: Path) -> None:
    """ROC and precision-recall for the four detectors, on the grouped test set."""
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    palette = {
        "isolation_forest": ANOMALY,
        "one_class_svm": HEALTHY,
        "pca_reconstruction": ACCENT,
        "mahalanobis": DARK,
    }
    for detector in detectors:
        scores = detector.anomaly_score(split.X_test)
        fpr, tpr, _ = roc_curve(split.y_test, scores)
        precision, recall, _ = precision_recall_curve(split.y_test, scores)
        axes[0].plot(
            fpr, tpr, linewidth=1.8, color=palette[detector.name], label=PRETTY[detector.name]
        )
        axes[1].plot(recall, precision, linewidth=1.8, color=palette[detector.name])

    axes[0].plot([0, 1], [0, 1], linestyle=":", color=GREY, linewidth=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right", fontsize=9)

    share = float(split.y_test.mean())
    axes[1].axhline(share, linestyle=":", color=GREY, linewidth=1)
    axes[1].text(0.02, share + 0.01, f"always anomaly ({share:.2f})", fontsize=8, color=GREY)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall")
    figure.suptitle("One-class detectors on 93 unseen executions, no failure seen in training")
    figure.tight_layout()
    figure.savefig(out / "roc-pr-curves.png")
    plt.close(figure)


def figure_confusion(split, detectors, out: Path) -> None:
    """Where the errors fall, for the best supervised model and the best detector."""
    forest = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            ),
        ]
    ).fit(split.X_train, split.y_train)
    detector = next(d for d in detectors if d.name == "mahalanobis")

    panels = [
        (
            "Random Forest, trained on labelled failures",
            confusion(split.y_test, forest.predict(split.X_test)),
        ),
        (
            "Mahalanobis, healthy runs only",
            confusion(split.y_test, detector.predict(split.X_test)),
        ),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for axis, (title, cells) in zip(axes, panels, strict=True):
        matrix = np.array(
            [
                [cells["true_negative"], cells["false_positive"]],
                [cells["false_negative"], cells["true_positive"]],
            ]
        )
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for i in range(2):
            for j in range(2):
                axis.text(
                    j,
                    i,
                    str(matrix[i][j]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="white" if matrix[i][j] > matrix.max() * 0.55 else DARK,
                )
        axis.set_xticks([0, 1], ["predicted healthy", "predicted failure"], fontsize=9)
        axis.set_yticks([0, 1], ["actually healthy", "actually failed"], fontsize=9)
        axis.set_title(title, fontsize=10)
        axis.grid(visible=False)
    figure.suptitle("Grouped held-out set, 93 executions, 67 of them failures")
    figure.tight_layout()
    figure.savefig(out / "confusion-matrices.png")
    plt.close(figure)


def figure_transfer(experiments: dict, out: Path) -> None:
    """Transfer to an unseen task phase, with and without removing duplicates."""
    disjoint = experiments.get("transfer", {})
    naive = experiments.get("transfer_naive", {})
    if not disjoint:
        logger.warning("no transfer study in the report, figure skipped")
        return

    subsets = sorted(disjoint)
    position = np.arange(len(subsets))
    width = 0.38

    def best(section, subset):
        values = [d["roc_auc"] for d in section[subset]["detectors"].values()]
        return float(np.mean(values))

    naive_values = [best(naive, subset) for subset in subsets] if naive else None
    disjoint_values = [best(disjoint, subset) for subset in subsets]

    figure, axis = plt.subplots(figsize=(8.6, 4.2))
    if naive_values:
        axis.bar(
            position - width / 2,
            naive_values,
            width,
            color=GREY,
            label="subset held out, its duplicates left in training",
        )
    axis.bar(
        position + width / 2 if naive_values else position,
        disjoint_values,
        width,
        color=HEALTHY,
        label="subset held out, every copy of its traces removed",
    )
    axis.set_xticks(position)
    axis.set_xticklabels(
        [
            f"{subset}\n{disjoint[subset]['n_train_healthy']} healthy runs left"
            for subset in subsets
        ],
        fontsize=9,
    )
    axis.set_ylim(0.8, 1.03)
    axis.set_ylabel("ROC-AUC, averaged over the four detectors")
    axis.set_title("Transfer to an unseen phase of the assembly task")
    axis.legend(loc="lower left", fontsize=9)
    figure.savefig(out / "transfer-across-subsets.png")
    plt.close(figure)


def figure_importance(split, feature_columns, out: Path) -> None:
    """Which sensor statistics the forest actually uses."""
    forest = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1).fit(
        split.X_train, split.y_train
    )
    order = np.argsort(forest.feature_importances_)[::-1][:15][::-1]

    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    axis.barh(
        np.arange(len(order)),
        forest.feature_importances_[order],
        color=[ANOMALY if feature_columns[i].startswith("F") else HEALTHY for i in order],
        height=0.66,
    )
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels([feature_columns[i] for i in order], fontsize=9)
    axis.set_xlabel("mean decrease in impurity")
    axis.set_title("The fifteen sensor statistics the forest leans on")
    axis.grid(axis="y", visible=False)
    axis.legend(
        handles=[
            Patch(color=ANOMALY, label="force channel"),
            Patch(color=HEALTHY, label="torque channel"),
        ],
        loc="lower right",
    )
    figure.savefig(out / "feature-importance.png")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("assets"))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()
    style()
    config = load_config(args.config)
    set_seed(config.random_state)
    args.output.mkdir(parents=True, exist_ok=True)

    benchmark = json.loads((args.reports / "benchmark.json").read_text())
    experiments = json.loads((args.reports / "experiments.json").read_text())

    frame, _ = encode_labels(load_robot_data(), binary=True)
    features = create_statistical_features(frame)
    X, columns = feature_matrix(features)
    y = frame["label_encoded"].to_numpy(dtype=int)
    split = grouped_split(X, y, trace_ids(frame), config.cv_folds, config.random_state)
    detectors = fit_detectors(split.X_train, split.y_train, config)

    figure_detection(split, detectors, args.output)
    figure_signals(frame, args.output)
    figure_duplicate_leak(benchmark, args.output)
    figure_comparison(benchmark, experiments, args.output)
    figure_threshold(experiments, args.output)
    figure_curves(split, detectors, args.output)
    figure_confusion(split, detectors, args.output)
    figure_transfer(experiments, args.output)
    figure_importance(split, columns, args.output)

    written = sorted(path.name for path in args.output.glob("*.png"))
    logger.info("%d figures written to %s: %s", len(written), args.output, ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
