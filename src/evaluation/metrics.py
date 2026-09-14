"""Metrics, threshold analysis and the controls that make a score believable.

Three things live here and they answer three different questions.

``score`` answers "how good is this prediction", with names that cannot drift.
The notebooks printed a weighted F1 in one place and the positive-class F1 in
another, for the same model; here ``f1_anomaly`` is the reference and
``f1_weighted`` is reported beside it under its own name.

``threshold_curve`` answers "how good is this *detector*, independently of
where the threshold was put". A detector that emits a continuous anomaly score
has a ROC-AUC and a PR-AUC that no threshold choice can flatter.

``shuffled_label_control`` answers "is this score real". A perfect score on 463
samples is an alarm, not a result. Destroy the signal and the model must follow
it down; if it does not, something in the features is carrying the answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


def score(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """One set of metrics, named without ambiguity.

    ``f1_anomaly`` is the reference: the positive class is the failure, and on
    a line that is the class whose recall has a cost. ``f1_weighted`` is
    reported next to it because that is what the original notebooks displayed,
    and on an imbalanced set the two differ by a wide margin.
    """
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_anomaly": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_anomaly": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_anomaly": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    # ROC-AUC reads well on a balanced set and flatters an imbalanced one.
    # PR-AUC is reported beside it because the deployment case, rare failures
    # among healthy runs, is the case where the two diverge most.
    result["roc_auc"] = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    result["pr_auc"] = average_precision_score(y_true, y_proba) if y_proba is not None else None
    return result


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """The confusion matrix as four named counts.

    A 2x2 array whose orientation the reader has to guess is a frequent source
    of inverted conclusions in anomaly detection write-ups, so the four cells
    are returned by name.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    (tn, fp), (fn, tp) = matrix
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


@dataclass(frozen=True)
class ThresholdAnalysis:
    """How a continuous anomaly score behaves across every possible threshold.

    Attributes:
        roc_auc: ranking quality, independent of any threshold.
        pr_auc: average precision, the honest summary when the positive class
            is not the rare one here but would be on a real line.
        best_f1: the highest positive-class F1 reachable by threshold choice.
        best_threshold: where that maximum sits.
        f1_at_percentiles: F1 obtained when the threshold is set at a given
            percentile of the *training* score distribution, which is what a
            deployment without labels would actually do.
    """

    roc_auc: float
    pr_auc: float
    best_f1: float
    best_threshold: float
    f1_at_percentiles: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "best_f1": self.best_f1,
            "best_threshold": self.best_threshold,
            "f1_at_percentiles": self.f1_at_percentiles,
        }


def threshold_curve(
    y_true: np.ndarray,
    anomaly_score: np.ndarray,
    train_score: np.ndarray | None = None,
    percentiles: tuple[int, ...] = (90, 95, 99),
) -> ThresholdAnalysis:
    """Evaluate a continuous anomaly score without committing to a threshold.

    Args:
        y_true: 1 for a failed execution, 0 for a healthy one.
        anomaly_score: higher means more anomalous. Sign conventions differ
            between scikit-learn detectors, so callers normalise before here.
        train_score: the same score computed on the healthy training runs. When
            given, thresholds are also placed at its percentiles, which is the
            only rule available when no labelled failures exist.
        percentiles: which percentiles of ``train_score`` to try.

    Returns:
        A :class:`ThresholdAnalysis`.
    """
    roc = float(roc_auc_score(y_true, anomaly_score))
    pr = float(average_precision_score(y_true, anomaly_score))

    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    best_index = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_index]) if len(thresholds) else float("nan")

    at_percentiles: dict[str, float] = {}
    if train_score is not None and len(train_score):
        for percentile in percentiles:
            cutoff = float(np.percentile(train_score, percentile))
            predicted = (anomaly_score > cutoff).astype(int)
            at_percentiles[f"p{percentile}"] = float(
                f1_score(y_true, predicted, pos_label=1, zero_division=0)
            )

    return ThresholdAnalysis(
        roc_auc=roc,
        pr_auc=pr,
        best_f1=float(np.max(f1[:-1])) if len(thresholds) else 0.0,
        best_threshold=best_threshold,
        f1_at_percentiles=at_percentiles,
    )


def shuffled_label_control(
    estimator, X: np.ndarray, y: np.ndarray, folds: int = 5, draws: int = 1, seed: int = 0
) -> dict:
    """Refit the model on permuted labels and check that it collapses.

    A leaking pipeline stays good even when the labels are noise, because the
    answer reaches the model through a channel the permutation does not touch.
    A clean one fits patterns that are not there and lands *below* the constant
    baseline. That gap is the evidence, and it is cheap to produce.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    real = float(cross_val_score(estimator, X, y, cv=cv, scoring="f1").mean())

    generator = np.random.default_rng(seed)
    shuffled = [
        float(cross_val_score(estimator, X, generator.permutation(y), cv=cv, scoring="f1").mean())
        for _ in range(draws)
    ]

    logger.info("real cv f1 %.4f, shuffled %.4f", real, float(np.mean(shuffled)))
    return {
        "cv_f1_real": real,
        "cv_f1_shuffled": float(np.mean(shuffled)),
        "draws": draws,
    }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
    model_name: str = "Model",
    show_confusion_matrix: bool = True,
    show_roc: bool = True,
    show_report: bool = True,
) -> dict:
    """
    Evaluate a classification model and display comprehensive metrics.

    This function computes and displays:
    - Accuracy, Precision, Recall, F1-score
    - Confusion matrix (visualization)
    - ROC curve and AUC (for binary classification)
    - Classification report (per-class metrics)

    Reference: Lesson 3 - Model Evaluation and Performance Metrics

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        y_pred_proba (Optional[np.ndarray]): Predicted class probabilities (for ROC curve).
                                           Shape should be (n_samples,) for binary classification.
                                           Default is None.
        model_name (str): Model name for display. Default is "Model".
        show_confusion_matrix (bool): If True, displays confusion matrix. Default is True.
        show_roc (bool): If True, displays ROC curve (requires y_pred_proba). Default is True.
        show_report (bool): If True, displays classification report. Default is True.

    Returns:
        Dict: Dictionary containing evaluation metrics:
            - 'accuracy': float
            - 'precision': float
            - 'recall': float
            - 'f1_score': float
            - 'auc': float (if y_pred_proba provided and binary classification)
            - 'model_name': str

    Example:
        >>> results = evaluate_model(y_test, y_pred, y_pred_proba, model_name='Random Forest')
        >>> print(results['f1_score'])
        0.95
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"Results for {model_name}")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Confusion matrix
    if show_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sorted(np.unique(y_true)),
            yticklabels=sorted(np.unique(y_true)),
            cbar_kws={"label": "Number of Instances"},
        )
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Predictions", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.tight_layout()
        plt.show()

    # ROC curve (only for binary classification)
    if show_roc and y_pred_proba is not None:
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

            print(f"AUC: {roc_auc:.4f}")
        else:
            print("⚠️  ROC curve available only for binary classification")

    # Classification report
    if show_report:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "model_name": model_name,
    }

    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        results["auc"] = roc_auc_score(y_true, y_pred_proba)

    return results
