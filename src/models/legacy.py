"""Supervised and unsupervised models for robot execution anomaly detection.

Authors: Ilyess Assadi, Adel Bousri.

Provides:
- supervised classifiers (logistic regression, random forest, SVM, gradient boosting)
- unsupervised detectors (isolation forest, one-class SVM)
- grid search with cross-validation, and model persistence

Note on the evaluation protocol. The trainers here fit their own scaler before
the model, which is what the notebooks used and what leaks the test set's
statistics when the caller has already split. `scripts/benchmark.py` supersedes
this module for anything published: it puts the scaler inside a Pipeline, so it
is refitted on the training folds alone. Prefer it when the numbers matter.
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

warnings.filterwarnings("ignore")


class SupervisedModels:
    """
    Class for managing supervised learning models with hyperparameter tuning.

    This class provides methods for training multiple supervised learning models
    with GridSearchCV for hyperparameter optimization and cross-validation.
    It automatically handles feature scaling where necessary and stores models,
    scalers, and evaluation results.

    Reference: Lesson 3 - Supervised Learning and Hyperparameter Tuning
               Lesson 4 - Cross-Validation and Model Selection

    Attributes:
        models (Dict): Dictionary storing trained models {model_name: model}
        best_params (Dict): Dictionary storing best hyperparameters {model_name: params}
        scalers (Dict): Dictionary storing StandardScalers {model_name: scaler}
        results (Dict): Dictionary storing evaluation results {model_name: metrics}
        cv_results (Dict): Dictionary storing cross-validation results {model_name: cv_scores}
    """

    def __init__(self):
        """Initialize SupervisedModels with empty dictionaries for models, parameters, and
        results."""
        self.models = {}
        self.best_params = {}
        self.scalers = {}
        self.results = {}
        self.cv_results = {}

    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict | None = None,
        cv: int = 5,
        n_jobs: int = -1,
        scoring: str = "f1_weighted",
    ) -> LogisticRegression:
        """
        Train Logistic Regression model with GridSearchCV and cross-validation.

        Logistic Regression is a linear classification model that learns a decision boundary
        by fitting a logistic function. It requires feature scaling for optimal performance.

        Reference: Lesson 3 - Linear Models and Regularization

        Args:
            X_train (np.ndarray): Training features (n_samples, n_features).
            y_train (np.ndarray): Training labels (n_samples,).
            param_grid (Optional[Dict]): Hyperparameter grid for GridSearchCV.
                                        Default includes C, penalty, solver, max_iter.
            cv (int): Number of cross-validation folds. Default is 5.
            n_jobs (int): Number of parallel jobs. Default is -1 (all cores).
            scoring (str): Scoring metric for GridSearchCV. Default is 'f1_weighted'.

        Returns:
            LogisticRegression: Best trained model from GridSearchCV.

        Example:
            >>> models = SupervisedModels()
            >>> lr = models.train_logistic_regression(X_train, y_train, cv=5)
        """
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "max_iter": [100, 200, 500],
            }

        print("Training Logistic Regression with GridSearchCV...")
        print(f"  - Cross-validation folds: {cv}")
        print(f"  - Scoring metric: {scoring}")

        # Standardize features (required for Logistic Regression)
        # StandardScaler ensures zero mean and unit variance, improving convergence
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers["logistic_regression"] = scaler

        # GridSearchCV with cross-validation
        # GridSearchCV performs exhaustive search over parameter grid
        # and evaluates each combination using K-fold cross-validation
        lr = LogisticRegression(random_state=42, n_jobs=n_jobs)
        grid_search = GridSearchCV(
            lr,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
        )
        grid_search.fit(X_train_scaled, y_train)

        # Store best model and parameters
        self.models["logistic_regression"] = grid_search.best_estimator_
        self.best_params["logistic_regression"] = grid_search.best_params_

        # Store cross-validation results
        self.cv_results["logistic_regression"] = {
            "mean_score": grid_search.best_score_,
            "std_score": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
            "all_scores": grid_search.cv_results_["mean_test_score"],
        }

        print(f"  - Best parameters: {grid_search.best_params_}")
        print(
            f"  - Best CV score: {grid_search.best_score_:.4f} ± "
            f"{self.cv_results['logistic_regression']['std_score']:.4f}"
        )

        return grid_search.best_estimator_

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict | None = None,
        cv: int = 5,
        n_jobs: int = -1,
        scoring: str = "f1_weighted",
    ) -> RandomForestClassifier:
        """
        Train Random Forest model with GridSearchCV and cross-validation.

        Random Forest is an ensemble method that combines multiple decision trees.
        It doesn't require feature scaling and provides feature importance analysis.
        Random Forest is robust to overfitting and handles non-linear relationships well.

        Reference: Lesson 3 - Ensemble Methods and Decision Trees

        Args:
            X_train (np.ndarray): Training features (n_samples, n_features).
            y_train (np.ndarray): Training labels (n_samples,).
            param_grid (Optional[Dict]): Hyperparameter grid for GridSearchCV.
                                        Default includes n_estimators, max_depth, min_samples_split.
            cv (int): Number of cross-validation folds. Default is 5.
            n_jobs (int): Number of parallel jobs. Default is -1 (all cores).
            scoring (str): Scoring metric for GridSearchCV. Default is 'f1_weighted'.

        Returns:
            RandomForestClassifier: Best trained model from GridSearchCV.

        Example:
            >>> models = SupervisedModels()
            >>> rf = models.train_random_forest(X_train, y_train, cv=5)
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

        print("Training Random Forest with GridSearchCV...")
        print(f"  - Cross-validation folds: {cv}")
        print(f"  - Scoring metric: {scoring}")

        # Random Forest doesn't require feature scaling
        # GridSearchCV with cross-validation
        rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
        )
        grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.models["random_forest"] = grid_search.best_estimator_
        self.best_params["random_forest"] = grid_search.best_params_

        # Store cross-validation results
        self.cv_results["random_forest"] = {
            "mean_score": grid_search.best_score_,
            "std_score": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
            "all_scores": grid_search.cv_results_["mean_test_score"],
        }

        print(f"  - Best parameters: {grid_search.best_params_}")
        print(
            f"  - Best CV score: {grid_search.best_score_:.4f} ± "
            f"{self.cv_results['random_forest']['std_score']:.4f}"
        )

        return grid_search.best_estimator_

    def train_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict | None = None,
        cv: int = 5,
        n_jobs: int = -1,
        sample_size: int | None = None,
        scoring: str = "f1_weighted",
    ) -> SVC:
        """
        Train Support Vector Machine (SVM) with GridSearchCV and cross-validation.

        SVM with RBF kernel can learn non-linear decision boundaries. It requires
        feature scaling for optimal performance. SVM can be computationally expensive
        for large datasets, so optional sampling is provided.

        Reference: Lesson 3 - Support Vector Machines and Kernel Methods

        Args:
            X_train (np.ndarray): Training features (n_samples, n_features).
            y_train (np.ndarray): Training labels (n_samples,).
            param_grid (Optional[Dict]): Hyperparameter grid for GridSearchCV.
                                        Default includes C, gamma, kernel.
            cv (int): Number of cross-validation folds. Default is 5.
            n_jobs (int): Number of parallel jobs. Default is -1 (all cores).
            sample_size (Optional[int]): If specified, samples this many instances
                                      for faster training. Useful for large datasets.
            scoring (str): Scoring metric for GridSearchCV. Default is 'f1_weighted'.

        Returns:
            SVC: Best trained model from GridSearchCV.

        Example:
            >>> models = SupervisedModels()
            >>> svm = models.train_svm(X_train, y_train, cv=5, sample_size=1000)
        """
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto", 0.001, 0.01],
                "kernel": ["rbf"],
            }

        print("Training SVM with GridSearchCV...")
        print(f"  - Cross-validation folds: {cv}")
        print(f"  - Scoring metric: {scoring}")

        # Optional sampling for large datasets (SVM can be slow)
        if sample_size and len(X_train) > sample_size:
            print(f"  - Sampling {sample_size} instances for faster training")
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train

        # Standardize features (required for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        self.scalers["svm"] = scaler

        # GridSearchCV with cross-validation
        # probability=True enables predict_proba for ROC curve calculation
        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(
            svm,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
        )
        grid_search.fit(X_train_scaled, y_train_subset)

        # Store best model and parameters
        self.models["svm"] = grid_search.best_estimator_
        self.best_params["svm"] = grid_search.best_params_

        # Store cross-validation results
        self.cv_results["svm"] = {
            "mean_score": grid_search.best_score_,
            "std_score": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
            "all_scores": grid_search.cv_results_["mean_test_score"],
        }

        print(f"  - Best parameters: {grid_search.best_params_}")
        print(
            f"  - Best CV score: {grid_search.best_score_:.4f} ± "
            f"{self.cv_results['svm']['std_score']:.4f}"
        )

        return grid_search.best_estimator_

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict | None = None,
        cv: int = 5,
        n_jobs: int = -1,
        scoring: str = "f1_weighted",
    ) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting model with GridSearchCV and cross-validation.

        Gradient Boosting is an ensemble method that builds models sequentially,
        where each model corrects errors of the previous one. It can achieve
        high performance but requires careful hyperparameter tuning to avoid overfitting.

        Reference: Lesson 3 - Ensemble Methods and Boosting

        Args:
            X_train (np.ndarray): Training features (n_samples, n_features).
            y_train (np.ndarray): Training labels (n_samples,).
            param_grid (Optional[Dict]): Hyperparameter grid for GridSearchCV.
                                        Default includes n_estimators, learning_rate, max_depth.
            cv (int): Number of cross-validation folds. Default is 5.
            n_jobs (int): Number of parallel jobs. Default is -1 (all cores).
            scoring (str): Scoring metric for GridSearchCV. Default is 'f1_weighted'.

        Returns:
            GradientBoostingClassifier: Best trained model from GridSearchCV.

        Example:
            >>> models = SupervisedModels()
            >>> gb = models.train_gradient_boosting(X_train, y_train, cv=5)
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5],
            }

        print("Training Gradient Boosting with GridSearchCV...")
        print(f"  - Cross-validation folds: {cv}")
        print(f"  - Scoring metric: {scoring}")

        # Gradient Boosting doesn't require feature scaling
        # GridSearchCV with cross-validation
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
        )
        grid_search.fit(X_train, y_train)

        # Store best model and parameters
        self.models["gradient_boosting"] = grid_search.best_estimator_
        self.best_params["gradient_boosting"] = grid_search.best_params_

        # Store cross-validation results
        self.cv_results["gradient_boosting"] = {
            "mean_score": grid_search.best_score_,
            "std_score": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
            "all_scores": grid_search.cv_results_["mean_test_score"],
        }

        print(f"  - Best parameters: {grid_search.best_params_}")
        print(
            f"  - Best CV score: {grid_search.best_score_:.4f} ± "
            f"{self.cv_results['gradient_boosting']['std_score']:.4f}"
        )

        return grid_search.best_estimator_

    def predict(self, model_name: str, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions using a trained model.

        This method automatically applies feature scaling if the model requires it
        (e.g., Logistic Regression, SVM). It handles both predictions and probability
        predictions for ROC curve calculation.

        Args:
            model_name (str): One of 'logistic_regression', 'random_forest',
                'svm', 'gradient_boosting').
            X (np.ndarray): Features to predict (n_samples, n_features).
            return_proba (bool): If True, also returns predicted probabilities. Default is False.

        Returns:
            np.ndarray: Predicted labels. With return_proba=True, a tuple
                (predictions, probabilities).

        Raises:
            ValueError: If model_name is not found in trained models.

        Example:
            >>> y_pred = models.predict('random_forest', X_test)
            >>> y_pred, y_proba = models.predict('random_forest', X_test, return_proba=True)
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(self.models.keys())}"
            )

        model = self.models[model_name]

        # Apply scaling if necessary (models that require scaling have scalers stored)
        if model_name in self.scalers:
            X = self.scalers[model_name].transform(X)

        if return_proba:
            if hasattr(model, "predict_proba"):
                return model.predict(X), model.predict_proba(X)[
                    :, 1
                ]  # Return probability of positive class
            else:
                return model.predict(X), None
        else:
            return model.predict(X)

    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model and its associated scaler to disk.

        This method saves the model, scaler (if applicable), and best hyperparameters
        to a pickle file for later use.

        Args:
            model_name (str): Name of the model to save.
            filepath (str): Path to save the model file.

        Raises:
            ValueError: If model_name is not found in trained models.

        Example:
            >>> models.save_model('random_forest', 'models/rf_model.pkl')
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model_data = {
            "model": self.models[model_name],
            "scaler": self.scalers.get(model_name),
            "best_params": self.best_params.get(model_name),
            "cv_results": self.cv_results.get(model_name),
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model {model_name} saved to {filepath}")


def train_supervised_model(
    model_type: str, X_train: np.ndarray, y_train: np.ndarray, **kwargs
) -> object:
    """
    Helper function to train a supervised learning model.

    This function provides a convenient interface for training models without
    explicitly creating a SupervisedModels instance.

    Reference: Lesson 3 - Supervised Learning Models

    Args:
        model_type (str): One of 'logistic_regression', 'random_forest',
            'svm', 'gradient_boosting'.
        X_train (np.ndarray): Training features (n_samples, n_features).
        y_train (np.ndarray): Training labels (n_samples,).
        **kwargs: Additional arguments passed to the training method.

    Returns:
        object: Trained model instance.

    Raises:
        ValueError: If model_type is not recognized.

    Example:
        >>> lr = train_supervised_model('logistic_regression', X_train, y_train, cv=5)
    """
    supervised_models = SupervisedModels()

    if model_type == "logistic_regression":
        return supervised_models.train_logistic_regression(X_train, y_train, **kwargs)
    elif model_type == "random_forest":
        return supervised_models.train_random_forest(X_train, y_train, **kwargs)
    elif model_type == "svm":
        return supervised_models.train_svm(X_train, y_train, **kwargs)
    elif model_type == "gradient_boosting":
        return supervised_models.train_gradient_boosting(X_train, y_train, **kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'logistic_regression', 'random_forest', 'svm', 'gradient_boosting'"
        )


def train_isolation_forest(
    X_train: np.ndarray, contamination: float = 0.1, n_estimators: int = 100, random_state: int = 42
) -> IsolationForest:
    """
    Train Isolation Forest for unsupervised anomaly detection.

    Isolation Forest is an unsupervised anomaly detection algorithm that isolates
    anomalies by randomly selecting features and splitting values. Anomalies are
    easier to isolate than normal points, making this method effective for detecting
    outliers in high-dimensional data.

    The contamination parameter must be between 0.0 and 0.5 (inclusive) or 'auto'.
    If contamination > 0.5, it's automatically capped at 0.5.

    Reference: Lesson 5 - Unsupervised Learning and Anomaly Detection

    Args:
        X_train (np.ndarray): Training features (n_samples, n_features).
                             Should contain only "normal" data for best results.
        contamination (float or str): Expected proportion of anomalies.
                                     Must be in range (0.0, 0.5] or 'auto'.
                                     Default is 0.1.
        n_estimators (int): Number of trees in the forest. Default is 100.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        IsolationForest: Trained Isolation Forest model.

    Example:
        >>> iso_forest = train_isolation_forest(X_train_normal, contamination='auto')
    """
    print("Training Isolation Forest...")

    # Validate and adjust contamination if necessary
    if contamination != "auto":
        if contamination <= 0.0:
            print(f"⚠️  Contamination ({contamination}) <= 0.0, using 'auto'")
            contamination = "auto"
        elif contamination > 0.5:
            print(f"⚠️  Contamination ({contamination:.4f}) > 0.5, capping at 0.5 (maximum allowed)")
            contamination = 0.5

    print(f"  - Contamination: {contamination}")
    print(f"  - Number of trees: {n_estimators}")

    # Isolation Forest doesn't require feature scaling, but it can help
    iso_forest = IsolationForest(
        contamination=contamination, n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )

    iso_forest.fit(X_train)

    print("✅ Isolation Forest trained")

    return iso_forest


def train_one_class_svm(
    X_train: np.ndarray,
    nu: float = 0.1,
    kernel: str = "rbf",
    gamma: str = "scale",
    random_state: int = 42,
) -> tuple[OneClassSVM, StandardScaler]:
    """
    Train One-Class SVM for unsupervised anomaly detection.

    One-Class SVM learns a decision boundary around the training data (normal instances).
    Points outside this boundary are classified as anomalies. It requires feature scaling
    for optimal performance, especially with RBF kernel.

    Reference: Lesson 5 - Unsupervised Learning and Anomaly Detection

    Args:
        X_train (np.ndarray): Training features (n_samples, n_features).
                             Should contain only "normal" data.
        nu (float): Upper bound on fraction of training errors and lower bound on
                   fraction of support vectors. Must be in range (0, 1]. Default is 0.1.
        kernel (str): Kernel type ('rbf', 'linear', 'poly'). Default is 'rbf'.
        gamma (str or float): Kernel coefficient. 'scale', 'auto', or float value.
                             Default is 'scale'.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        Tuple[OneClassSVM, StandardScaler]: Trained model and its scaler; the
            scaler is required at prediction time.

    Example:
        >>> oc_svm, scaler = train_one_class_svm(X_train_normal, nu=0.1, kernel='rbf')
        >>> predictions = oc_svm.predict(scaler.transform(X_test))
    """
    print("Training One-Class SVM...")
    print(f"  - Nu: {nu}")
    print(f"  - Kernel: {kernel}")
    print(f"  - Gamma: {gamma}")

    # Standardize features (required for One-Class SVM, especially with RBF kernel)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # One-Class SVM doesn't have a random_state parameter, but we document it for consistency
    oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    oc_svm.fit(X_train_scaled)

    print("✅ One-Class SVM trained")

    return oc_svm, scaler


def compare_models(
    results_dict: dict[str, dict], metric: str = "accuracy", figsize: tuple[int, int] = (12, 6)
) -> pd.DataFrame:
    """
    Compare performance of multiple models using a bar chart and summary table.

    This function creates a visual comparison of model performance and returns
    a DataFrame with all metrics for further analysis.

    Reference: Lesson 4 - Model Comparison and Selection

    Args:
        results_dict (Dict[str, Dict]): Dictionary of results {model_name: {metric: value}}.
        metric (str): Metric to compare, such as 'accuracy', 'f1_score' or
            'auc'. Default is 'accuracy'.
        figsize (Tuple[int, int]): Figure size (width, height). Default is (12, 6).

    Returns:
        pd.DataFrame: DataFrame with model comparison (models as rows, metrics as columns).

    Example:
        >>> results = {
        ...     'Logistic Regression': {'accuracy': 0.89, 'f1_score': 0.92},
        ...     'Random Forest': {'accuracy': 0.93, 'f1_score': 0.95}
        ... }
        >>> comparison_df = compare_models(results, metric='accuracy')
    """
    model_names = list(results_dict.keys())
    metric_values = [results_dict[name].get(metric, 0) for name in model_names]

    # Bar chart comparison
    plt.figure(figsize=figsize)
    colors = ["steelblue", "forestgreen", "coral", "purple", "orange", "brown"]
    plt.bar(
        model_names, metric_values, color=colors[: len(model_names)], edgecolor="black", alpha=0.7
    )
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f"Model Comparison - {metric.capitalize()}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    if metric_values:
        max_val = max(metric_values)
        for i, v in enumerate(metric_values):
            plt.text(i, v + max_val * 0.01, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show()

    # Comparison table
    comparison_df = pd.DataFrame(results_dict).T
    print("\nModel Comparison:")
    print(comparison_df.to_string())

    return comparison_df
