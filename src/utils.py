"""Backward-compatible surface for the notebooks, plus shared paths.

The code that used to live here now sits in ``src/data``, ``src/features``,
``src/evaluation`` and ``src/visualization``. The names are re-exported at their
old address so that notebooks 01 to 05, which are kept as the record of what
the project looked like before the audit, keep running unchanged.

New code should import from the packages directly.
"""

from __future__ import annotations

from .config import ASSETS_DIR, REPORTS_DIR, ROOT, Config, load_config, set_seed, setup_logging
from .data.loader import (
    DATA_DIR,
    HEALTHY_LABELS,
    META_COLUMNS,
    SAMPLES_PER_EXECUTION,
    SENSOR_NAMES,
    as_time_series,
    encode_labels,
    load_robot_data,
    raw_columns,
)
from .evaluation.metrics import (
    confusion,
    evaluate_model,
    score,
    shuffled_label_control,
    threshold_curve,
)
from .features.statistical import (
    N_STATISTICAL_FEATURES,
    create_statistical_features,
    feature_matrix,
    feature_names,
)
from .visualization.plots import (
    plot_class_distribution,
    plot_correlations,
    plot_feature_importances,
    plot_pca,
    plot_time_series,
)

__all__ = [
    "ASSETS_DIR",
    "DATA_DIR",
    "HEALTHY_LABELS",
    "META_COLUMNS",
    "N_STATISTICAL_FEATURES",
    "REPORTS_DIR",
    "ROOT",
    "SAMPLES_PER_EXECUTION",
    "SENSOR_NAMES",
    "Config",
    "as_time_series",
    "confusion",
    "create_statistical_features",
    "encode_labels",
    "evaluate_model",
    "feature_matrix",
    "feature_names",
    "load_config",
    "load_robot_data",
    "plot_class_distribution",
    "plot_correlations",
    "plot_feature_importances",
    "plot_pca",
    "plot_time_series",
    "raw_columns",
    "score",
    "set_seed",
    "setup_logging",
    "shuffled_label_control",
    "threshold_curve",
]
