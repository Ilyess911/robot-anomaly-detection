"""Figures for exploration and for the report.

Two families live here. The first is exploratory and is called from the
notebooks: class distribution, correlations, a single execution's traces, a PCA
projection, feature importances. The second is in ``scripts/figures.py``, which
regenerates every image the README publishes so that no figure can outlive the
numbers it illustrates. That separation exists because the original figures
were committed before a labelling fix and quietly kept showing the old story.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

#: Palette used across every figure of the project, so that a colour means the
#: same thing from one image to the next: healthy is blue, failure is red.
HEALTHY_COLOR = "#2E5E8C"
ANOMALY_COLOR = "#C0392B"
NEUTRAL_COLOR = "#7F8C8D"


def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
    source_col: str | None = None,
    title: str = "Class Distribution",
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Visualize class distribution in the dataset.

    This function creates two visualizations:
    1. Bar chart showing class counts
    2. Stacked bar chart showing class distribution by source (if available)

    Reference: Lesson 1 - Exploratory Data Analysis and Class Imbalance

    Args:
        df (pd.DataFrame): DataFrame containing class labels.
        label_col (str): Name of column containing class labels. Default is 'label'.
        source_col (Optional[str]): Name of column containing data source (LP1-LP5).
                                   If None, uses 'source' if available. Default is None.
        title (str): Title for the plot. Default is "Class Distribution".
        figsize (Tuple[int, int]): Figure size (width, height). Default is (12, 6).

    Example:
        >>> plot_class_distribution(df, label_col='label_binary',
        ...                        title='Binary Class Distribution')
    """
    _fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Class distribution bar chart
    class_counts = df[label_col].value_counts()
    axes[0].bar(
        range(len(class_counts)),
        class_counts.values,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].set_xticks(range(len(class_counts)))
    axes[0].set_xticklabels(class_counts.index, rotation=45, ha="right")
    axes[0].set_xlabel("Class", fontsize=12)
    axes[0].set_ylabel("Number of Instances", fontsize=12)
    axes[0].set_title("Distribution by Class", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Distribution by source (if available)
    source_col = source_col or "source"
    if source_col in df.columns:
        source_counts = df.groupby([source_col, label_col]).size().unstack(fill_value=0)
        source_counts.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set3")
        axes[1].set_xlabel("Source (LP1-LP5)", fontsize=12)
        axes[1].set_ylabel("Number of Instances", fontsize=12)
        axes[1].set_title("Distribution by Source and Class", fontsize=14, fontweight="bold")
        axes[1].legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        # If no source column, hide second subplot
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nDistribution Statistics:")
    print(f"  - Total instances: {len(df)}")
    print(f"  - Number of classes: {len(class_counts)}")
    print("  - Distribution:")
    for label, count in class_counts.items():
        print(f"    {label}: {count} ({count / len(df) * 100:.2f}%)")


def plot_correlations(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    feature_columns: list[str] | None = None,
    figsize: tuple[int, int] = (15, 12),
    sample_size: int | None = None,
) -> None:
    """
    Visualize correlation matrix between features using a heatmap.

    This function computes pairwise correlations between features and displays them
    as a heatmap. For large feature sets (e.g., 90 features), it's recommended to
    sample features to improve visualization clarity and computation speed.

    Reference: Lesson 2 - Feature Analysis and Correlation

    Args:
        df (pd.DataFrame): DataFrame containing features.
        feature_cols (Optional[List[str]]): List of feature column names to analyze.
                                           If None, uses all numeric columns. Default is None.
        feature_columns (Optional[List[str]]): Alias for feature_cols (for compatibility).
                                              Default is None.
        figsize (Tuple[int, int]): Figure size (width, height). Default is (15, 12).
        sample_size (int | None): If given, keeps this many features, chosen
            by variance.
            Recommended for large feature sets (90+ features). Default is None.

    Example:
        >>> plot_correlations(df, sample_size=30)  # Analyze top 30 most variant features
    """
    # Use feature_columns if provided (for compatibility)
    if feature_columns is not None:
        feature_cols = feature_columns

    if feature_cols is None:
        # Exclude non-numeric columns
        feature_cols = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]
        feature_cols = [
            col
            for col in feature_cols
            if col not in ["label", "label_encoded", "label_binary", "source", "label_original"]
        ]

    # Sample features if requested (for performance and clarity)
    if sample_size and len(feature_cols) > sample_size:
        # Select features with highest variance (most informative)
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(sample_size).index.tolist()
        print(
            f"⚠️  Using {sample_size} of {len(df.columns)} features "
            f"(highest variance) for faster computation"
        )

    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr()

    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()

    # Print correlation statistics
    # Extract upper triangle (excluding diagonal) for statistics
    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    print("\nCorrelation Statistics:")
    print(f"  - Number of features: {len(feature_cols)}")
    print(f"  - Mean correlation: {upper_triangle.mean():.3f}")
    print(f"  - Max correlation: {upper_triangle.max():.3f}")
    print(f"  - Min correlation: {upper_triangle.min():.3f}")


def plot_time_series(
    df: pd.DataFrame,
    instance_idx: int = 0,
    sensors: list[str] | None = None,
    samples_per_sensor: int = 15,
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """
    Visualize time-series sensor readings for a specific instance.

    This function reshapes the 90 features (15 samples × 6 sensors) into a 2D array
    and plots time-series for each sensor. This visualization helps understand temporal
    patterns in sensor data, which is crucial for anomaly detection.

    Reference: Lesson 1 - Time-Series Data Visualization

    Args:
        df (pd.DataFrame): DataFrame with features (90 features = 15 samples × 6 sensors).
        instance_idx (int): Index of instance to visualize. Default is 0.
        sensors (List[str]): List of sensor names. Default is ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'].
        samples_per_sensor (int): Number of time samples per sensor. Default is 15.
        figsize (Tuple[int, int]): Figure size (width, height). Default is (15, 10).

    Example:
        >>> plot_time_series(df, instance_idx=0,
        ...                  title='Normal Operation Sensor Readings')
    """
    if sensors is None:
        sensors = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    if instance_idx >= len(df):
        print(f"⚠️  Index {instance_idx} out of bounds (max: {len(df) - 1})")
        return

    # Extract features for this instance (exclude label, source, etc.)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    instance_data = df.iloc[instance_idx][feature_cols].values

    # Reshape data: 15 samples × 6 sensors
    if len(instance_data) == samples_per_sensor * len(sensors):
        data_reshaped = instance_data.reshape(samples_per_sensor, len(sensors))
        time_steps = np.arange(1, samples_per_sensor + 1)

        # Create subplots for each sensor
        _fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for i, sensor in enumerate(sensors):
            axes[i].plot(time_steps, data_reshaped[:, i], marker="o", linewidth=2, markersize=4)
            axes[i].set_xlabel("Time Sample", fontsize=10)
            axes[i].set_ylabel(f"{sensor}", fontsize=10)
            axes[i].set_title(f"Sensor {sensor}", fontsize=12, fontweight="bold")
            axes[i].grid(alpha=0.3)

        plt.suptitle(
            f"Time-Series Sensor Readings - Instance {instance_idx}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.show()

        # Display instance label and source
        if "label" in df.columns:
            label = df.iloc[instance_idx]["label"]
            source = df.iloc[instance_idx].get("source", "Unknown")
            print(f"\nInstance {instance_idx}:")
            print(f"  Label: {label}")
            print(f"  Source: {source}")
    else:
        print(
            f"⚠️  Unrecognised data format. Expected "
            f"{samples_per_sensor * len(sensors)} features, "
            f"got: {len(instance_data)}"
        )


def plot_pca(
    X: np.ndarray,
    y: np.ndarray | None = None,
    n_components: int = 2,
    title: str = "PCA Visualization",
    figsize: tuple[int, int] = (12, 8),
) -> PCA:
    """
    Visualize data using Principal Component Analysis (PCA).

    This function applies PCA dimensionality reduction and visualizes the result
    in 2D or 3D space. PCA helps identify patterns and clusters in high-dimensional
    data by projecting it onto principal components that capture maximum variance.

    Reference: Lesson 2 - Principal Component Analysis and Dimensionality Reduction

    Args:
        X (np.ndarray): Feature matrix (n_samples, n_features).
        y (Optional[np.ndarray]): Class labels for coloring points. Default is None.
        n_components (int): Number of principal components (2 or 3). Default is 2.
        title (str): Plot title. Default is "PCA Visualization".
        figsize (Tuple[int, int]): Figure size (width, height). Default is (12, 8).

    Returns:
        PCA: Fitted PCA object with explained variance information.

    Example:
        >>> pca = plot_pca(X_scaled, y=y, n_components=2, title='PCA: Normal vs Anomaly')
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Visualization
    if n_components == 2:
        plt.figure(figsize=figsize)
        if y is not None:
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6, s=50)
            plt.colorbar(scatter, label="Class")
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)", fontsize=12)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    elif n_components == 3:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enregistre la projection 3d)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            if y is not None:
                scatter = ax.scatter(
                    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap="viridis", alpha=0.6, s=50
                )
                plt.colorbar(scatter, label="Class")
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.6, s=50)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)", fontsize=10)
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)", fontsize=10)
            ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2] * 100:.2f}%)", fontsize=10)
            ax.set_title(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("⚠️  mpl_toolkits.mplot3d not available. 3D visualization skipped.")

    # Print explained variance
    print("\nExplained Variance:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {var * 100:.2f}%")
    print(f"  Total: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

    return pca


def plot_feature_importances(
    model,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importances",
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Visualize feature importances from a trained model.

    This function extracts feature importances from tree-based models (Random Forest,
    Gradient Boosting) or coefficients from linear models (Logistic Regression) and
    displays them in a bar chart. Feature importance analysis helps understand which
    sensors or features are most critical for anomaly detection.

    Reference: Lesson 3 - Model Interpretability and Feature Importance

    Args:
        model: Trained model with 'feature_importances_' attribute (tree-based) or
              'coef_' attribute (linear models).
        feature_names (List[str]): List of feature names corresponding to model features.
        top_n (int): Number of top features to display. Default is 20.
        title (str): Plot title. Default is "Feature Importances".
        figsize (Tuple[int, int]): Figure size (width, height). Default is (12, 8).

    Example:
        >>> plot_feature_importances(rf_model, feature_names=stat_feature_cols,
        ...                         title='Random Forest Feature Importances')
    """
    # Extract importances
    if hasattr(model, "feature_importances_"):
        # Tree-based models (Random Forest, Gradient Boosting)
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models (Logistic Regression)
        # Take absolute value of coefficients for importance
        importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
    else:
        print("⚠️  Model does not have 'feature_importances_' or 'coef_' attribute")
        return

    # Create DataFrame
    importance_df = (
        pd.DataFrame({"feature": feature_names[: len(importances)], "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    # Visualization
    plt.figure(figsize=figsize)
    plt.barh(
        range(len(importance_df)),
        importance_df["importance"],
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.yticks(range(len(importance_df)), importance_df["feature"])
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nTop {top_n} Most Important Features:")
    for _i, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
