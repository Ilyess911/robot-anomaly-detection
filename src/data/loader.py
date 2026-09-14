"""Parsing and labelling of the UCI Robot Execution Failures dataset.

The dataset stores one execution as a label line followed by fifteen lines of
six force and torque readings. This module turns that layout into a flat frame
of 90 raw samples per execution, and encodes the label.

The single most important object in this file is ``HEALTHY_LABELS``. LP3 names
its healthy class ``ok`` and never uses ``normal``; matching on ``normal``
alone labelled 20 healthy executions as failures and made that subset look
entirely faulty. See ``docs/data-quality-audit.md``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

#: Labels that denote a healthy execution, across every subset. ``ok`` is LP3's,
#: which never uses ``normal``. Any label outside this set is a failure.
HEALTHY_LABELS = frozenset({"normal", "ok"})

#: Bookkeeping columns. They travel with the data but never enter a model:
#: ``label_encoded`` and ``label_binary`` *are* the target, and letting them in
#: buys a perfect score for the worst possible reason.
META_COLUMNS = frozenset({"label", "label_encoded", "label_binary", "label_original", "source"})

#: The six force and torque channels of the wrist sensor, in file order.
SENSOR_NAMES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")

#: Time steps recorded per execution.
SAMPLES_PER_EXECUTION = 15

#: The repository's data directory, resolved from this module's own location.
#: A relative path would depend on the working directory, which differs between
#: a notebook, a script and an interactive session.
DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")

#: The five subsets, each a phase of the same assembly task.
SUBSET_FILES = {
    "LP1": "lp1.data",
    "LP2": "lp2.data",
    "LP3": "lp3.data",
    "LP4": "lp4.data",
    "LP5": "lp5.data",
}


def _parse_subset(path: str, source: str) -> list[dict]:
    """Parse one ``.data`` file into a list of execution records.

    The format has no separator other than the shape of the lines: a label is a
    single token on its own line, and the fifteen lines that follow hold six
    integers each, tab or space separated. An execution is only accepted once
    the fifteen lines have been read, so a truncated record is skipped rather
    than silently padded.
    """
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]

    executions: list[dict] = []
    index = 0
    while index < len(lines):
        label = lines[index]
        if len(label.split()) != 1:
            index += 1
            continue

        readings: list[float] = []
        for line in lines[index + 1 : index + 1 + SAMPLES_PER_EXECUTION]:
            values = [value for value in line.replace("\t", " ").split() if value]
            if len(values) != len(SENSOR_NAMES):
                break
            try:
                readings.extend(float(value) for value in values)
            except ValueError:
                break

        expected = SAMPLES_PER_EXECUTION * len(SENSOR_NAMES)
        if len(readings) != expected:
            index += 1
            continue

        record = {"label": label, "source": source}
        record.update({f"feature_{i + 1}": value for i, value in enumerate(readings)})
        executions.append(record)
        index += SAMPLES_PER_EXECUTION + 1

    return executions


def load_robot_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load and merge the five subsets into one frame.

    Args:
        data_dir: directory holding ``lp1.data`` to ``lp5.data``. Defaults to
            the repository's own ``data/``, resolved from this module's path so
            that the call works from a notebook, a script or a REPL alike.

    Returns:
        A frame of 463 rows: 90 raw sensor columns (``feature_1`` to
        ``feature_90``), the original ``label``, and the ``source`` subset.

    Raises:
        FileNotFoundError: if no subset could be read.

    Example:
        >>> load_robot_data().shape
        (463, 92)
    """
    executions: list[dict] = []

    for source, filename in SUBSET_FILES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            logger.warning("subset %s missing at %s", source, path)
            continue

        parsed = _parse_subset(path, source)
        if not parsed:
            logger.warning("subset %s parsed to zero executions", source)
            continue

        executions.extend(parsed)
        logger.info("%s: %d executions", source, len(parsed))

    if not executions:
        raise FileNotFoundError(f"no usable .data file under {data_dir}")

    frame = pd.DataFrame(executions)
    columns = [column for column in frame.columns if column.startswith("feature_")]
    columns.sort(key=lambda name: int(name.split("_")[1]))

    logger.info(
        "loaded %d executions, %d raw columns, %d distinct labels",
        len(frame),
        len(columns),
        frame["label"].nunique(),
    )
    return frame[[*columns, "label", "source"]].copy()


def encode_labels(df: pd.DataFrame, binary: bool = False) -> tuple[pd.DataFrame, dict]:
    """Encode the class label, either as healthy/failure or as the 16 classes.

    Binary encoding is the one the published results use: healthy is 0, every
    failure label is 1, and membership is decided by ``HEALTHY_LABELS`` rather
    than by an equality against the string ``normal``.

    Counting from the raw files: 109 ``normal`` plus 20 ``ok`` is 129 healthy
    executions out of 463, so 72% failures rather than the 76% that matching on
    ``normal`` alone produced. Every score computed under that mistake is
    affected, starting with the majority-class baseline.

    Args:
        df: frame carrying a ``label`` column.
        binary: healthy against failure when true, all 16 classes otherwise.

    Returns:
        The frame with ``label_encoded`` added (and ``label_binary`` in the
        binary case), plus the mapping from code to label.
    """
    encoded = df.copy()

    if "label_original" not in encoded.columns:
        encoded["label_original"] = encoded["label"].copy()

    if binary:
        encoded["label_binary"] = encoded["label"].apply(
            lambda value: 0 if str(value).lower() in HEALTHY_LABELS else 1
        )
        encoded["label_encoded"] = encoded["label_binary"]
        mapping = {0: "normal", 1: "anomaly"}
    else:
        label_encoder = LabelEncoder()
        encoded["label_encoded"] = label_encoder.fit_transform(encoded["label"])
        mapping = dict(
            zip(
                label_encoder.classes_,
                label_encoder.transform(label_encoder.classes_),
                strict=False,
            )
        )

    logger.info("labels encoded: %s", mapping)
    return encoded, mapping


def raw_columns(df: pd.DataFrame) -> list[str]:
    """The 90 raw sensor columns, in acquisition order."""
    columns = [column for column in df.columns if column.startswith("feature_")]
    return sorted(columns, key=lambda name: int(name.split("_")[1]))


def as_time_series(df: pd.DataFrame) -> np.ndarray:
    """Reshape the raw columns into ``(n_executions, 15 steps, 6 sensors)``.

    The statistical features collapse the time axis; the detectors that work on
    residuals and the plots that show a signal need it back.
    """
    values = df[raw_columns(df)].to_numpy(dtype=float)
    return values.reshape(len(df), SAMPLES_PER_EXECUTION, len(SENSOR_NAMES))


def trace_ids(df: pd.DataFrame, decimals: int = 6) -> np.ndarray:
    """Give every distinct sensor trace an identifier, shared by its copies.

    The five subsets are not five recording sessions. LP2 and LP3 annotate the
    same 47 executions under two different fault taxonomies, LP4 and LP5 share
    116 executions, and the merged frame therefore holds 463 rows over 251
    distinct traces. A random split puts the same physical execution on both
    sides of the line.

    Grouping by trace is what stops that. The identifier is the rounded vector
    of the 90 raw readings, so two rows share an identifier exactly when they
    are the same recording.

    Args:
        df: frame carrying the raw sensor columns.
        decimals: rounding applied before comparison. The files store integers,
            so this only guards against a float round-trip.

    Returns:
        An integer array of length ``len(df)``, one group per distinct trace.
    """
    values = df[raw_columns(df)].to_numpy(dtype=float).round(decimals)
    keys = [value.tobytes() for value in values]
    lookup: dict[bytes, int] = {}
    return np.array([lookup.setdefault(key, len(lookup)) for key in keys], dtype=int)


def duplicate_summary(df: pd.DataFrame) -> dict:
    """How much of the merged dataset is the same execution counted twice.

    Returned rather than printed because it belongs in the published report:
    this is the number that decides whether a score on this dataset means
    anything.
    """
    groups = trace_ids(df)
    _, counts = np.unique(groups, return_counts=True)

    binary = df["label_binary"].to_numpy() if "label_binary" in df.columns else None
    conflicting = 0
    if binary is not None:
        for group in np.unique(groups):
            if len(set(binary[groups == group])) > 1:
                conflicting += 1

    return {
        "n_executions": len(df),
        "n_distinct_traces": len(counts),
        "n_traces_once": int((counts == 1).sum()),
        "n_traces_twice": int((counts == 2).sum()),
        "n_traces_three_or_more": int((counts > 2).sum()),
        "duplicate_share": float(1 - len(counts) / len(df)),
        "traces_with_conflicting_binary_label": int(conflicting),
    }
