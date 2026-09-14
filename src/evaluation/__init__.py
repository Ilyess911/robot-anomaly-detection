"""Evaluation: metrics, threshold analysis, and the leak controls."""

from .metrics import (
    ThresholdAnalysis,
    confusion,
    evaluate_model,
    score,
    shuffled_label_control,
    threshold_curve,
)
from .protocol import Split, cv_splitter, grouped_split, random_split

__all__ = [
    "Split",
    "ThresholdAnalysis",
    "confusion",
    "cv_splitter",
    "evaluate_model",
    "grouped_split",
    "random_split",
    "score",
    "shuffled_label_control",
    "threshold_curve",
]
