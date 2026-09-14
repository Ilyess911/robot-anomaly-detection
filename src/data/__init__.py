"""Dataset access: parsing, labelling, and the raw time-series view."""

from .loader import (
    DATA_DIR,
    HEALTHY_LABELS,
    META_COLUMNS,
    SAMPLES_PER_EXECUTION,
    SENSOR_NAMES,
    SUBSET_FILES,
    as_time_series,
    duplicate_summary,
    encode_labels,
    load_robot_data,
    raw_columns,
    trace_ids,
)

__all__ = [
    "DATA_DIR",
    "HEALTHY_LABELS",
    "META_COLUMNS",
    "SAMPLES_PER_EXECUTION",
    "SENSOR_NAMES",
    "SUBSET_FILES",
    "as_time_series",
    "duplicate_summary",
    "encode_labels",
    "load_robot_data",
    "raw_columns",
    "trace_ids",
]
