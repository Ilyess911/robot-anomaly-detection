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
    "always_anomaly": 'Always answer "failure"',
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
    figure.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10))
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
    axis.set_ylabel("cross-validated F1, failure class")
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

    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    axis.barh(np.arange(len(values)), values, color=colors, height=0.66)
    for index, value in enumerate(values):
        axis.text(value + 0.006, index, f"{value:.3f}", va="center", fontsize=9)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_xlim(0, 1.09)
    axis.set_xlabel("F1 on the failure class, grouped held-out set of 93 executions")
    axis.set_title("Baselines, one-class detectors, supervised classifiers")
    axis.grid(axis="y", visible=False)
    handles = [
        Patch(color=GREY, label="baseline, not a model"),
        Patch(color=ACCENT, label="one-class, healthy runs only"),
        Patch(color=HEALTHY, label="supervised, labelled failures"),
    ]
    # Below the axis: every horizontal bar reaches past 0.65, so any in-axes
    # legend sits on top of data.
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        fontsize=9,
    )
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
    axis.set_ylabel("F1, failure class")
    axis.set_title("Every detector ranks perfectly. Only the threshold separates them")
    axis.legend(loc="lower left", ncol=2)
    figure.savefig(out / "threshold-sensitivity.png")
    plt.close(figure)


def figure_cross_validated(experiments: dict, out: Path) -> None:
    """The headline one-class numbers with their uncertainty, over 25 fits.

    A single fold put all four detectors at ROC-AUC 1.000 and invited exactly
    the disbelief that deserves. Five repeats of grouped five-fold give an
    interval instead of a point, and the intervals are what decide whether two
    detectors differ at all.
    """
    cv = experiments.get("cross_validated", {})
    if not cv:
        logger.warning("no cross-validated section in the report, figure skipped")
        return

    names = [name for name in DETECTOR_ORDER if name in cv]
    position = np.arange(len(names))
    width = 0.38

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    for axis, metric, title in (
        (axes[0], "roc_auc", "Ranking quality (ROC-AUC)"),
        (axes[1], "f1_percentile", "F1 at the label-free threshold"),
    ):
        means = [cv[name][metric]["mean"] for name in names]
        errors = [
            [cv[name][metric]["mean"] - cv[name][metric]["ci95_low"] for name in names],
            [cv[name][metric]["ci95_high"] - cv[name][metric]["mean"] for name in names],
        ]
        axis.bar(position, means, width * 1.6, yerr=errors, capsize=4, color=HEALTHY)
        axis.set_xticks(position)
        axis.set_xticklabels([PRETTY[name].replace(" ", "\n") for name in names], fontsize=8)
        axis.set_ylim(0.4, 1.03)
        axis.set_title(title, fontsize=11)
        axis.grid(axis="x", visible=False)

    axes[0].set_ylabel("mean over 25 folds, with 95% interval")
    figure.suptitle(
        "Five repeats of grouped five-fold. The single-fold 1.000 does not survive it", y=1.0
    )
    figure.tight_layout()
    figure.savefig(out / "cross-validated.png")
    plt.close(figure)


def figure_calibration(experiments: dict, out: Path) -> None:
    """What the threshold promised against what it delivered.

    The target is 5%. Three of the four detectors flag more than half of the
    healthy executions they have never seen, and the duplicates were hiding most
    of that: without grouping the same rule looks three times better than it is.
    """
    grouped = experiments.get("cross_validated", {})
    ungrouped = experiments.get("cross_validated_ungrouped", {})
    if not grouped:
        logger.warning("no cross-validated section in the report, figure skipped")
        return

    names = [name for name in DETECTOR_ORDER if name in grouped]
    position = np.arange(len(names))
    width = 0.26

    figure, axis = plt.subplots(figsize=(9.4, 4.6))
    series = [
        ("random folds, duplicates on both sides", ungrouped, "far_percentile", GREY),
        ("grouped folds, percentile rule", grouped, "far_percentile", ANOMALY),
        ("grouped folds, tolerance bound", grouped, "far_tolerance", HEALTHY),
    ]

    for index, (label, source, metric, color) in enumerate(series):
        if not source:
            continue
        means = [source[name][metric]["mean"] * 100 for name in names]
        errors = [
            [
                (source[name][metric]["mean"] - source[name][metric]["ci95_low"]) * 100
                for name in names
            ],
            [
                (source[name][metric]["ci95_high"] - source[name][metric]["mean"]) * 100
                for name in names
            ],
        ]
        axis.bar(
            position + (index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=color,
            label=label,
        )

    # The target is drawn as a legend entry rather than floating text: every
    # bar group is tall enough that an annotation lands on top of data.
    axis.axhline(
        5,
        color=DARK,
        linestyle="--",
        linewidth=1.2,
        label="target: 5% of healthy executions flagged",
    )
    axis.set_xticks(position)
    axis.set_xticklabels([PRETTY[name] for name in names], fontsize=9)
    axis.set_ylabel("false alarm rate actually realised (%)")
    axis.set_title("A threshold that promises 5% and delivers 60%")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    axis.grid(axis="x", visible=False)
    figure.savefig(out / "calibration.png")
    plt.close(figure)


def figure_deployment_cost(deployment: dict, out: Path) -> None:
    """Expected cost per execution on a line that fails once in a hundred.

    Four policies on one axis. Doing nothing and stopping on everything bound the
    problem; the best reachable operating point says what the detector is worth;
    the two label-free rules say what a deployment would actually get. The gap
    between the last two and the third is the price of not being able to
    calibrate.
    """
    economics = deployment.get("economics", {})
    if not economics:
        logger.warning("no economics section in the report, figure skipped")
        return

    names = [name for name in DETECTOR_ORDER if name in economics]
    reference = economics[names[0]]["reference_case"]
    nothing = reference["best_reachable"]["cost_of_doing_nothing"]
    always = reference["best_reachable"]["cost_of_always_stopping"]

    position = np.arange(len(names))
    width = 0.27

    figure, axis = plt.subplots(figsize=(9.6, 4.6))
    bars = [
        (
            "percentile rule, what a deployment gets",
            [
                economics[name]["reference_case"]["at_label_free_threshold"]["percentile"][
                    "expected_cost"
                ]
                for name in names
            ],
            ANOMALY,
        ),
        (
            "tolerance bound, what a guarantee gets",
            [
                economics[name]["reference_case"]["at_label_free_threshold"]["tolerance"][
                    "expected_cost"
                ]
                for name in names
            ],
            ACCENT,
        ),
        (
            "best reachable, needs the labels",
            [
                economics[name]["reference_case"]["best_reachable"]["expected_cost"]
                for name in names
            ],
            HEALTHY,
        ),
    ]

    for index, (label, values, color) in enumerate(bars):
        axis.bar(position + (index - 1) * width, values, width, color=color, label=label)

    # The two trivial policies land within a percent of each other here, so one
    # annotation carries both rather than two labels fighting for the same strip.
    axis.axhline(nothing, color=DARK, linestyle="--", linewidth=1.1)
    axis.axhline(always, color=GREY, linestyle=":", linewidth=1.1)
    axis.text(
        0.012,
        0.905,
        f"ignore every alarm ({nothing:.2f}) and stop on everything ({always:.2f})",
        transform=axis.transAxes,
        fontsize=8,
        color=DARK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )

    axis.set_xticks(position)
    axis.set_xticklabels([PRETTY[name] for name in names], fontsize=9)
    axis.set_ylabel("expected cost per execution, in false alarms")
    axis.set_title("One failure per hundred executions, a missed failure worth a hundred stops")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    axis.grid(axis="x", visible=False)
    figure.savefig(out / "deployment-cost.png")
    plt.close(figure)


def figure_score_distributions(split, detectors, out: Path) -> None:
    """Why four detectors that rank identically well score so differently.

    A ROC curve would be four overlapping right angles here, since every
    detector reaches 1.000, and it would say nothing. What separates them is
    where their alarm threshold lands inside their own score distribution, so
    that is what this draws: every score divided by its detector's threshold, on
    a log axis, with the alarm at 1.0 on all four panels.

    Read it as: anything above the line is flagged. A detector whose healthy
    test scores sit above 1.0 raises false alarms even though its ranking is
    perfect, and a detector whose failures sit below it misses them.

    One-Class SVM's decision function goes negative well inside its boundary, so
    a ratio there is negative and meaningless on a log axis. Those points are
    clipped to the bottom of the panel rather than dropped, which keeps the
    count honest at the cost of a flat row of markers.
    """
    figure, axes = plt.subplots(1, 4, figsize=(12.4, 4.0), sharey=True)
    generator = np.random.default_rng(0)

    for axis, detector in zip(axes, detectors, strict=True):
        groups = [
            ("healthy\ntraining", detector.train_scores_, HEALTHY),
            ("healthy\nheld out", detector.anomaly_score(split.X_test[split.y_test == 0]), ACCENT),
            ("failed\nheld out", detector.anomaly_score(split.X_test[split.y_test == 1]), ANOMALY),
        ]
        for position, (_label, values, color) in enumerate(groups):
            ratio = np.clip(values / detector.threshold_, 1e-3, None)
            jitter = generator.uniform(-0.16, 0.16, size=len(ratio))
            axis.scatter(position + jitter, ratio, s=13, color=color, alpha=0.7, zorder=3)

        axis.axhline(1.0, color=DARK, linestyle="--", linewidth=1.1, zorder=2)
        axis.set_yscale("log")
        axis.set_xticks(range(3), [group[0] for group in groups], fontsize=8)
        axis.set_xlim(-0.6, 2.6)
        axis.set_title(PRETTY[detector.name], fontsize=10)
        axis.grid(axis="x", visible=False)

    axes[0].set_ylabel("anomaly score, as a multiple of the alarm threshold")
    figure.suptitle(
        "Every detector ranks perfectly. The alarm line lands in a different place in each",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(out / "score-distributions.png")
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

    figure, axis = plt.subplots(figsize=(8.6, 4.6))
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
        [f"{subset}\n({disjoint[subset]['n_train_healthy']} healthy left)" for subset in subsets],
        fontsize=9,
    )
    axis.set_ylim(0.8, 1.03)
    axis.set_ylabel("ROC-AUC, averaged over the four detectors")
    axis.set_title("Transfer to an unseen phase of the assembly task")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9)
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
    deployment_path = args.reports / "deployment.json"
    deployment = json.loads(deployment_path.read_text()) if deployment_path.exists() else {}

    frame, _ = encode_labels(load_robot_data(), binary=True)
    features = create_statistical_features(frame)
    X, columns = feature_matrix(features)
    y = frame["label_encoded"].to_numpy(dtype=int)
    split = grouped_split(X, y, trace_ids(frame), config.cv_folds, config.random_state)
    detectors = fit_detectors(split.X_train, split.y_train, config)

    figure_detection(split, detectors, args.output)
    figure_cross_validated(experiments, args.output)
    figure_calibration(experiments, args.output)
    figure_deployment_cost(deployment, args.output)
    figure_signals(frame, args.output)
    figure_duplicate_leak(benchmark, args.output)
    figure_comparison(benchmark, experiments, args.output)
    figure_threshold(experiments, args.output)
    figure_score_distributions(split, detectors, args.output)
    figure_confusion(split, detectors, args.output)
    figure_transfer(experiments, args.output)
    figure_importance(split, columns, args.output)

    written = sorted(path.name for path in args.output.glob("*.png"))
    logger.info("%d figures written to %s: %s", len(written), args.output, ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
