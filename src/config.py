"""Experiment configuration, loaded from ``configs/*.toml``.

Parameters that can move a published number live in a file, not in a constant
scattered across three scripts. Each report records the configuration it ran
under, so a table in the README can be traced back to the protocol that
produced it.

TOML is read by the standard library from Python 3.11, so this adds no
dependency between the data and the result.
"""

from __future__ import annotations

import logging
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = ROOT / "assets"


@dataclass(frozen=True)
class Config:
    """The parameters of one experimental run."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    scoring: str = "f1"
    threshold_percentile: float = 95.0
    pca_variance: float = 0.95
    one_class_svm_nu: float = 0.1
    isolation_forest_trees: int = 200
    shuffled_label_draws: int = 5
    transfer_enabled: bool = True
    source: str = "defaults"
    raw: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """The configuration as it should appear inside a report."""
        return {
            "source": self.source,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "scoring": self.scoring,
            "threshold_percentile": self.threshold_percentile,
            "pca_variance": self.pca_variance,
            "one_class_svm_nu": self.one_class_svm_nu,
            "isolation_forest_trees": self.isolation_forest_trees,
            "shuffled_label_draws": self.shuffled_label_draws,
        }


def load_config(path: Path | str | None = None) -> Config:
    """Read a configuration file, falling back to the defaults above."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        logger.warning("no configuration at %s, using built-in defaults", path)
        return Config()

    raw = tomllib.loads(path.read_text())
    protocol = raw.get("protocol", {})
    detectors = raw.get("detectors", {})
    controls = raw.get("controls", {})
    transfer = raw.get("transfer", {})

    return Config(
        random_state=protocol.get("random_state", 42),
        test_size=protocol.get("test_size", 0.2),
        cv_folds=protocol.get("cv_folds", 5),
        scoring=protocol.get("scoring", "f1"),
        threshold_percentile=detectors.get("threshold_percentile", 95.0),
        pca_variance=detectors.get("pca_variance", 0.95),
        one_class_svm_nu=detectors.get("one_class_svm_nu", 0.1),
        isolation_forest_trees=detectors.get("isolation_forest_trees", 200),
        shuffled_label_draws=controls.get("shuffled_label_draws", 5),
        transfer_enabled=transfer.get("enabled", True),
        source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        raw=raw,
    )


def set_seed(seed: int) -> None:
    """Seed every generator this project can reach.

    scikit-learn estimators take ``random_state`` explicitly, which is the only
    reliable way. This covers what is left: the shuffles and permutations that
    reach for a global generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.debug("seed set to %d", seed)


def setup_logging(level: int = logging.INFO) -> None:
    """One log format for every script in the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
