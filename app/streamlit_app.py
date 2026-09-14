"""A browser demo: pick a detector, move the threshold, watch the alarms move.

The point is not to look like a product. It is to make the one result that a
table cannot convey visible in a few seconds: these detectors rank failures
almost perfectly, and almost all of the remaining error comes from *where the
alarm threshold is placed*, which is the one choice a deployment has to make
without labels.

Run it with:

    pip install -r requirements-lock.txt -r requirements-demo.txt
    streamlit run app/streamlit_app.py

Nothing published depends on this file. It is excluded from the benchmark, from
the reports and from the reproducibility check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Streamlit executes the script on a worker thread, and matplotlib's default
# macOS backend refuses to build a figure outside the main thread. Selecting the
# non-interactive backend has to happen before pyplot is imported, which is why
# it sits above the import rather than inside a function.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.loader import (
    SENSOR_NAMES,
    SUBSET_FILES,
    as_time_series,
    encode_labels,
    load_robot_data,
    trace_ids,
)
from src.evaluation.metrics import score
from src.features.statistical import create_statistical_features, feature_matrix
from src.models.detectors import DETECTORS

HEALTHY = "#2E6F9E"
ANOMALY = "#C0392B"
DARK = "#1B222B"

PRETTY = {
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "pca_reconstruction": "PCA reconstruction error",
    "mahalanobis": "Mahalanobis distance",
}


@st.cache_data
def load_everything():
    """Parse, label and featurise once per session."""
    frame, _ = encode_labels(load_robot_data(), binary=True)
    X, _columns = feature_matrix(create_statistical_features(frame))
    return frame, X, _columns, frame["label_encoded"].to_numpy(dtype=int), trace_ids(frame)


@st.cache_resource
def fit(detector_name: str, percentile: float, healthy_key: bytes, _healthy: np.ndarray):
    """Fit a detector, cached on the healthy data it was given.

    The array is passed with a leading underscore so Streamlit does not try to
    hash it, and ``healthy_key`` carries its bytes as the cache key instead.
    Without that split the cache either refuses the argument or silently keys on
    object identity.
    """
    config = load_config()
    keyword = {
        "isolation_forest": {"n_estimators": config.isolation_forest_trees},
        "one_class_svm": {"nu": config.one_class_svm_nu},
        "pca_reconstruction": {"variance": config.pca_variance},
        "mahalanobis": {},
    }[detector_name]
    return DETECTORS[detector_name](
        percentile=percentile, random_state=config.random_state, **keyword
    ).fit(_healthy)


def main() -> None:
    st.set_page_config(page_title="Robot anomaly detection", layout="wide")
    st.title("Anomaly detection on robot execution data")
    st.caption(
        "Six force and torque channels, fifteen time steps per execution. "
        "The detector is fitted on healthy runs only and has never seen a failure."
    )

    frame, X, _columns, y, groups = load_everything()

    with st.sidebar:
        st.header("Setup")
        subset = st.selectbox("Score this phase of the task", sorted(SUBSET_FILES), index=0)
        detector_name = st.selectbox(
            "Detector", list(PRETTY), format_func=lambda name: PRETTY[name]
        )
        percentile = st.slider(
            "Alarm threshold, as a percentile of the healthy training scores",
            min_value=80.0,
            max_value=99.5,
            value=95.0,
            step=0.5,
        )
        st.markdown(
            "Training excludes every copy of the scored traces. LP2 and LP3 "
            "annotate the same 47 recordings and LP4 and LP5 share 116 more, so "
            "removing a subset by name alone would leave its executions behind."
        )

    target_mask = frame["source"].to_numpy() == subset
    train_mask = ~np.isin(groups, np.unique(groups[target_mask]))
    healthy = X[train_mask & (y == 0)]

    if len(healthy) < 20:
        st.error(f"only {len(healthy)} healthy executions left to fit on, which is too few")
        return

    detector = fit(detector_name, percentile, healthy.tobytes(), healthy)
    scores = detector.anomaly_score(X[target_mask])
    truth = y[target_mask]
    flagged = (scores > detector.threshold_).astype(int)
    measured = score(truth, flagged, scores)

    left, middle, right, far = st.columns(4)
    left.metric("Executions scored", len(scores))
    middle.metric("Flagged", int(flagged.sum()))
    right.metric("F1, failure class", f"{measured['f1_anomaly']:.3f}")
    far.metric("ROC-AUC", f"{measured['roc_auc']:.3f}")
    st.caption(
        f"Fitted on {len(healthy)} healthy executions from the other phases. "
        f"Recall {measured['recall_anomaly']:.3f}, precision {measured['precision_anomaly']:.3f}."
    )

    order = np.argsort(scores)
    figure, axis = plt.subplots(figsize=(10, 3.6))
    index = np.arange(len(scores))
    ordered_truth = truth[order]
    axis.scatter(
        index[ordered_truth == 0],
        scores[order][ordered_truth == 0],
        s=28,
        color=HEALTHY,
        label="healthy",
    )
    axis.scatter(
        index[ordered_truth == 1],
        scores[order][ordered_truth == 1],
        s=28,
        color=ANOMALY,
        label="failed",
    )
    axis.axhline(detector.threshold_, color=DARK, linestyle="--", linewidth=1.2, label="alarm")
    axis.set_yscale("log")
    axis.set_xlabel(f"{subset} executions, sorted by anomaly score")
    axis.set_ylabel("anomaly score")
    axis.legend(loc="upper left", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    st.pyplot(figure, width="stretch")

    st.subheader("Inspect one execution")
    positions = np.flatnonzero(target_mask)
    choice = st.selectbox(
        "Execution",
        range(len(positions)),
        format_func=lambda i: (
            f"#{i}  score {scores[i]:9.2f}  "
            f"{'flagged' if flagged[i] else 'not flagged'}  "
            f"recorded as {frame['label'].to_numpy()[positions][i]}"
        ),
    )
    traces = as_time_series(frame)[positions]
    healthy_mean = as_time_series(frame)[(y == 0)].mean(axis=0)

    figure, axes = plt.subplots(2, 3, figsize=(11, 4.6), sharex=True)
    for channel, (axis, sensor) in enumerate(zip(axes.ravel(), SENSOR_NAMES, strict=True)):
        axis.plot(healthy_mean[:, channel], color=HEALTHY, linewidth=1.6, label="healthy mean")
        axis.plot(
            traces[choice][:, channel],
            color=ANOMALY if flagged[choice] else DARK,
            linewidth=1.8,
            label="this execution",
        )
        axis.set_title(sensor, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")

    st.subheader("Every execution, worst first")
    st.dataframe(
        pd.DataFrame(
            {
                "execution": np.arange(len(scores)),
                "anomaly score": scores,
                "flagged": flagged.astype(bool),
                "recorded label": frame["label"].to_numpy()[positions],
            }
        ).sort_values("anomaly score", ascending=False),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
