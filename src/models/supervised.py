"""Supervised classifiers and the search grids behind the published numbers.

The grids are defined here rather than inside the benchmark script so that the
tests, the figures and the benchmark all search the same space. A grid that
differs between the script that publishes a number and the test that guards it
is a number nobody is guarding.

Every estimator is wrapped in a Pipeline whose first step is the scaler. That
placement is the whole point: inside a cross-validated search, the scaler is
refitted on the training folds alone, so no statistic of the held-out fold ever
reaches the model. The leaky variant, kept for comparison, standardises before
the split instead, which is the defect the original notebooks carried.
"""

from __future__ import annotations

import logging

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


def build_grids(random_state: int = 42) -> dict:
    """The four classifiers and the hyperparameters searched for each.

    The grids match the ones the course notebooks used, with one deliberate
    change: logistic regression's iteration floor is raised. The notebooks
    stopped at 100 iterations and produced hundreds of convergence warnings, and
    a model that has not converged is not a result, it is an incident.
    """
    return {
        "logistic_regression": (
            LogisticRegression(random_state=random_state),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear", "saga"],
                "model__max_iter": [500, 2000],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [10, 20, None],
            },
        ),
        "svm_rbf": (
            SVC(kernel="rbf", probability=True, random_state=random_state),
            {
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", "auto", 0.001, 0.01],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
            },
        ),
    }


def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    """Wrap an estimator, with the scaler inside the pipeline when asked.

    ``scale=False`` reproduces the leaky protocol, where the caller has already
    standardised the whole dataset before splitting it.
    """
    steps = (
        [("scaler", StandardScaler()), ("model", estimator)] if scale else [("model", estimator)]
    )
    return Pipeline(steps)


def search(
    estimator,
    grid: dict,
    X,
    y,
    folds: int = 5,
    random_state: int = 42,
    scale: bool = True,
    groups=None,
):
    """Grid search a classifier, scored on the positive-class F1.

    ``scoring="f1"`` is the anomaly class, not a weighted average. Selecting on
    the weighted average would reward a model that protects the majority class,
    which on this dataset is the failures and on a real line would be exactly
    backwards.

    ``groups`` carries the trace identity of each training row. When given, the
    folds are grouped, so a duplicated execution cannot sit in the fold that
    selects the hyperparameters and in the fold that scores them.
    """
    cv = (
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
        if groups is not None
        else StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    )
    fitted = GridSearchCV(
        build_pipeline(estimator, scale=scale), grid, cv=cv, scoring="f1", n_jobs=-1, refit=True
    )
    fitted.fit(X, y, groups=groups) if groups is not None else fitted.fit(X, y)
    return fitted


def decision_stump(random_state: int = 42) -> DecisionTreeClassifier:
    """One feature, one threshold. The baseline almost nobody publishes."""
    return DecisionTreeClassifier(max_depth=1, random_state=random_state)


def shallow_tree(random_state: int = 42) -> DecisionTreeClassifier:
    """Depth two, so three decisions in total."""
    return DecisionTreeClassifier(max_depth=2, random_state=random_state)
