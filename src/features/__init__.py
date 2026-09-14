"""Feature engineering: from raw sensor traces to model-ready descriptors."""

from .statistical import (
    N_STATISTICAL_FEATURES,
    STATISTICS,
    create_statistical_features,
    feature_matrix,
    feature_names,
)

__all__ = [
    "N_STATISTICAL_FEATURES",
    "STATISTICS",
    "create_statistical_features",
    "feature_matrix",
    "feature_names",
]
