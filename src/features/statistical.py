"""Statistical descriptors extracted from each execution's sensor traces.

Ninety raw samples on 463 executions invites overfitting and resists
interpretation. Eight statistics per sensor cut the space to 48 dimensions and
produce names a maintenance engineer can argue with: ``Tz_std`` is the
variability of yaw torque, not "feature 74".

The eight, per channel:

===========  =======================================================
mean         central tendency of the channel over the execution
std          variability, the closest thing here to vibration energy
min / max    extremes, where a collision leaves its mark
range        max minus min, the amplitude of the event
skew         asymmetry of the distribution of readings
kurtosis     tail weight, heavy on short violent transients
trend        slope of a linear fit over the 15 steps, so drift
===========  =======================================================
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..data.loader import META_COLUMNS, SAMPLES_PER_EXECUTION, SENSOR_NAMES

logger = logging.getLogger(__name__)

#: The eight statistics computed per sensor, in the order they appear in the
#: feature names. 6 sensors x 8 statistics = the 48 columns the models see.
STATISTICS = ("mean", "std", "min", "max", "range", "skew", "kurtosis", "trend")

N_STATISTICAL_FEATURES = len(SENSOR_NAMES) * len(STATISTICS)


def feature_names() -> list[str]:
    """The 48 feature names, in column order."""
    return [f"{sensor}_{statistic}" for sensor in SENSOR_NAMES for statistic in STATISTICS]


def _describe_channel(values: np.ndarray) -> dict[str, float]:
    """The eight statistics for one channel of one execution.

    Degenerate channels are handled explicitly rather than left to produce
    ``nan``: a flat signal has zero skew and zero kurtosis by convention here,
    and a failed polynomial fit contributes a zero slope rather than poisoning
    the whole row.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    minimum, maximum = float(np.min(values)), float(np.max(values))
    deviation = float(np.std(values))

    if deviation == 0 or len(np.unique(values)) == 1:
        skew = kurtosis = 0.0
    else:
        skew = float(scipy_stats.skew(values))
        kurtosis = float(scipy_stats.kurtosis(values))
        skew = 0.0 if np.isnan(skew) else skew
        kurtosis = 0.0 if np.isnan(kurtosis) else kurtosis

    try:
        trend = float(np.polyfit(range(len(values)), values, 1)[0])
        trend = 0.0 if np.isnan(trend) else trend
    except (np.linalg.LinAlgError, ValueError):
        trend = 0.0

    return {
        "mean": float(np.mean(values)),
        "std": deviation,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "skew": skew,
        "kurtosis": kurtosis,
        "trend": trend,
    }


def create_statistical_features(
    df: pd.DataFrame,
    samples_per_sensor: int = SAMPLES_PER_EXECUTION,
    num_sensors: int = len(SENSOR_NAMES),
) -> pd.DataFrame:
    """Turn the 90 raw samples of each execution into 48 statistical features.

    Args:
        df: frame holding ``feature_1`` to ``feature_90`` and the bookkeeping
            columns.
        samples_per_sensor: time steps per execution, 15 in this dataset.
        num_sensors: channels per time step, 6 in this dataset.

    Returns:
        A frame of 48 feature columns plus whichever bookkeeping columns the
        input carried. Those bookkeeping columns include the target, so callers
        must drop them by name before fitting; ``META_COLUMNS`` names them.

    Example:
        >>> create_statistical_features(frame).shape
        (463, 53)
    """
    expected = samples_per_sensor * num_sensors
    raw = [column for column in df.columns if column.startswith("feature_")]
    raw.sort(key=lambda name: int(name.split("_")[1]))

    if len(raw) != expected:
        raise ValueError(f"expected {expected} raw sensor columns, found {len(raw)}")

    traces = df[raw].to_numpy(dtype=np.float64).reshape(len(df), samples_per_sensor, num_sensors)

    rows = []
    for trace in traces:
        row: dict[str, float] = {}
        for index, sensor in enumerate(SENSOR_NAMES[:num_sensors]):
            for statistic, value in _describe_channel(trace[:, index]).items():
                row[f"{sensor}_{statistic}"] = value
        rows.append(row)

    features = pd.DataFrame(rows, columns=feature_names())
    for column in META_COLUMNS:
        if column in df.columns:
            features[column] = df[column].to_numpy()

    logger.info("%d statistical features over %d executions", len(feature_names()), len(features))
    return features


def feature_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Split a feature frame into the model matrix and its column names.

    Bookkeeping columns are excluded by name and never by dtype.
    ``label_encoded`` and ``label_binary`` are numeric and they *are* the
    target: a dtype filter lets them through and returns a perfect score.
    """
    columns = [column for column in features.columns if column not in META_COLUMNS]
    if len(columns) != N_STATISTICAL_FEATURES:
        raise RuntimeError(f"expected {N_STATISTICAL_FEATURES} features, found {len(columns)}")
    return features[columns].to_numpy(dtype=float), columns
