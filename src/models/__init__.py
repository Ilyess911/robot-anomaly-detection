"""Models: supervised classifiers and one-class detectors.

``legacy`` holds the trainers written for the course notebooks. They fit their
own scaler before the model, which leaks when the caller has already split, and
they are kept only so that the notebooks still run. Nothing published comes
from them; ``supervised`` and ``detectors`` do that work.
"""

from .detectors import (
    DETECTORS,
    BaseDetector,
    IsolationForestDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
    build_detectors,
)
from .legacy import (
    SupervisedModels,
    compare_models,
    train_isolation_forest,
    train_one_class_svm,
    train_supervised_model,
)
from .supervised import build_grids, build_pipeline, decision_stump, search, shallow_tree

__all__ = [
    "DETECTORS",
    "BaseDetector",
    "IsolationForestDetector",
    "MahalanobisDetector",
    "OneClassSVMDetector",
    "PCAReconstructionDetector",
    "SupervisedModels",
    "build_detectors",
    "build_grids",
    "build_pipeline",
    "compare_models",
    "decision_stump",
    "search",
    "shallow_tree",
    "train_isolation_forest",
    "train_one_class_svm",
    "train_supervised_model",
]
