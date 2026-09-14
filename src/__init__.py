"""An anomaly detection framework for robotic and industrial sensor data.

Layout:

===================  ====================================================
``src.data``         parsing and labelling of the raw executions
``src.features``     statistical descriptors per sensor
``src.models``       supervised classifiers and one-class detectors
``src.evaluation``   metrics, threshold analysis, leak controls
``src.visualization``figures for the notebooks and for the report
``src.config``       protocol parameters, seeding, logging
===================  ====================================================
"""

from .config import Config, load_config, set_seed, setup_logging
from .data import as_time_series, encode_labels, load_robot_data
from .evaluation import confusion, score, shuffled_label_control, threshold_curve
from .features import create_statistical_features, feature_matrix, feature_names
from .models import build_detectors, build_grids

__all__ = [
    "Config",
    "as_time_series",
    "build_detectors",
    "build_grids",
    "confusion",
    "create_statistical_features",
    "encode_labels",
    "feature_matrix",
    "feature_names",
    "load_config",
    "load_robot_data",
    "score",
    "set_seed",
    "setup_logging",
    "shuffled_label_control",
    "threshold_curve",
]
