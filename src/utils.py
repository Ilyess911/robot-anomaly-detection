"""
Utility module for data loading, preprocessing, and visualization for industrial robot anomaly detection.
Author: Alexandre Dupont

This module provides functions for:
- Loading and parsing UCI Robot Execution Failures dataset
- Data preprocessing and feature engineering
- Statistical feature extraction from time-series sensor data
- Visualization utilities for exploratory data analysis
- Model evaluation and performance metrics

Reference: Lesson 1 - Data Exploration and Preprocessing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


#: Répertoire des données du dépôt, résolu depuis l'emplacement de ce module.
#: Un chemin relatif dépendrait du dossier courant, qui n'est pas le même selon
#: qu'on lance un notebook, un script ou une session interactive.
DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")


def load_robot_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load and merge all .data files from the UCI Robot Execution Failures dataset.
    
    This function parses the unique file format where each instance consists of:
    - Line 1: Class label (e.g., "normal", "collision", "obstruction")
    - Lines 2-16: 15 time samples, each containing 6 sensor values (Fx, Fy, Fz, Tx, Ty, Tz)
    - Empty lines for separation between instances
    - Total: 90 features per instance (15 samples × 6 sensors)
    
    The function handles tab-separated and space-separated values, validates data integrity,
    and merges all sources (LP1-LP5) into a unified DataFrame.
    
    Reference: Lesson 1 - Data Collection and Integration
    
    Args:
        data_dir (str): Path to directory containing .data files. Defaults to the
            repository's own ``data/`` directory, resolved from this module's location
            so that the call works from a notebook, a script or a REPL alike.
        
    Returns:
        pd.DataFrame: Merged DataFrame with:
            - 90 feature columns (feature_1 to feature_90)
            - 'label' column: Original class labels
            - 'source' column: Data source identifier (LP1-LP5)
            
    Raises:
        FileNotFoundError: If no valid data files are found.
        
    Example:
        >>> df = load_robot_data("data")
        >>> print(df.shape)
        (463, 92)
        >>> print(df['source'].unique())
        ['LP1', 'LP2', 'LP3', 'LP4', 'LP5']
    """
    data_files = {
        'LP1': 'lp1.data',
        'LP2': 'lp2.data',
        'LP3': 'lp3.data',
        'LP4': 'lp4.data',
        'LP5': 'lp5.data'
    }
    
    all_instances = []
    
    print("Loading robot execution failure data...")
    print("="*60)
    
    for source, filename in data_files.items():
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue
        
        try:
            # Read file line by line, removing empty lines
            # This approach handles the unique file format where labels and data are interleaved
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            instances = []
            i = 0
            samples_per_instance = 15
            sensors_per_sample = 6
            
            # Parse file: each instance starts with a label, followed by 15 data lines
            while i < len(lines):
                # Current line should be a label (single word, non-numeric)
                label = lines[i]
                
                # Validate that it's a label (single word)
                if len(label.split()) != 1:
                    i += 1
                    continue
                
                # Read the 15 following lines of sensor data
                instance_data = []
                data_lines_read = 0
                
                for j in range(i + 1, min(i + 1 + samples_per_instance, len(lines))):
                    line = lines[j]
                    # Handle both tab-separated and space-separated values
                    if '\t' in line:
                        values = line.split('\t')
                    else:
                        values = line.split()
                    
                    # Clean empty values
                    values = [v.strip() for v in values if v.strip()]
                    
                    # Validate that it's a data line (exactly 6 sensor values)
                    if len(values) == sensors_per_sample:
                        try:
                            # Convert to numeric values (float)
                            numeric_values = [float(v) for v in values]
                            instance_data.extend(numeric_values)
                            data_lines_read += 1
                        except (ValueError, TypeError):
                            # If conversion fails, this might be a new label
                            break
                    elif len(values) == 0:
                        # Empty line, continue
                        continue
                    else:
                        # If not 6 values, this might be a new label
                        break
                
                # Validate that we have exactly 15 data lines (90 features)
                if data_lines_read == samples_per_instance:
                    # Create dictionary for this instance
                    instance = {
                        'label': label,
                        'source': source
                    }
                    
                    # Add all 90 features
                    for k in range(len(instance_data)):
                        instance[f'feature_{k+1}'] = instance_data[k]
                    
                    instances.append(instance)
                    i += data_lines_read + 1  # +1 for the label
                else:
                    # If we don't have 15 lines, move to next line
                    i += 1
            
            if instances:
                all_instances.extend(instances)
                print(f"✅ {source}: {len(instances)} instances loaded")
            else:
                print(f"⚠️  {source}: No valid instances found")
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_instances:
        raise FileNotFoundError("No data files found!")
    
    # Create DataFrame
    result_df = pd.DataFrame(all_instances)
    
    # Reorder columns: features first, then label and source
    feature_cols = [col for col in result_df.columns if col.startswith('feature_')]
    feature_cols.sort(key=lambda x: int(x.split('_')[1]))  # Sort by feature number
    
    # Create final DataFrame with correct column order
    final_df = result_df[feature_cols + ['label', 'source']].copy()
    
    print("="*60)
    print(f"✅ Total: {len(final_df)} instances")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Sources: {final_df['source'].unique().tolist()}")
    print(f"   Unique labels: {final_df['label'].unique().tolist()}")
    
    return final_df


def encode_labels(df: pd.DataFrame, binary: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode class labels into numeric format.
    
    This function supports two encoding strategies:
    - Binary encoding: Converts all labels to binary (normal=0, anomaly=1)
    - Multi-class encoding: Uses LabelEncoder for all distinct classes
    
    Reference: Lesson 1 - Target Variable Encoding
    
    Args:
        df (pd.DataFrame): DataFrame with 'label' column containing class labels.
        binary (bool): If True, performs binary encoding (normal vs anomaly).
                      If False, performs multi-class encoding. Default is False.
        
    Returns:
        Tuple[pd.DataFrame, Dict]:
            - DataFrame with encoded labels in 'label_encoded' column
            - Dictionary mapping encoded values to original labels
            
    Example:
        >>> df_encoded, mapping = encode_labels(df, binary=True)
        >>> print(mapping)
        {0: 'normal', 1: 'anomaly'}
    """
    df_encoded = df.copy()
    label_mapping = {}
    
    # Preserve original labels for reference
    if 'label_original' not in df_encoded.columns:
        df_encoded['label_original'] = df_encoded['label'].copy()
    
    if binary:
        # Binary encoding: normal=0, all other labels=1 (anomaly)
        # This simplifies the problem to binary classification for anomaly detection
        df_encoded['label_binary'] = df_encoded['label'].apply(
            lambda x: 0 if str(x).lower() == 'normal' else 1
        )
        df_encoded['label_encoded'] = df_encoded['label_binary']
        label_mapping = {0: 'normal', 1: 'anomaly'}
    else:
        # Multi-class encoding: preserve all distinct classes
        label_encoder = LabelEncoder()
        df_encoded['label_encoded'] = label_encoder.fit_transform(df_encoded['label'])
        label_mapping = dict(zip(
            label_encoder.classes_, 
            label_encoder.transform(label_encoder.classes_)
        ))
        # For multi-class, label_binary is not created
    
    print(f"Labels encoded: {label_mapping}")
    
    return df_encoded, label_mapping


def plot_class_distribution(df: pd.DataFrame, 
                           label_col: str = 'label',
                           source_col: Optional[str] = None,
                           title: str = "Class Distribution",
                           figsize: Tuple[int, int] = (12, 6)) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Class distribution bar chart
    class_counts = df[label_col].value_counts()
    axes[0].bar(range(len(class_counts)), class_counts.values, 
                color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xticks(range(len(class_counts)))
    axes[0].set_xticklabels(class_counts.index, rotation=45, ha='right')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Number of Instances', fontsize=12)
    axes[0].set_title('Distribution by Class', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Distribution by source (if available)
    source_col = source_col if source_col else 'source'
    if source_col in df.columns:
        source_counts = df.groupby([source_col, label_col]).size().unstack(fill_value=0)
        source_counts.plot(kind='bar', stacked=True, ax=axes[1], colormap='Set3')
        axes[1].set_xlabel('Source (LP1-LP5)', fontsize=12)
        axes[1].set_ylabel('Number of Instances', fontsize=12)
        axes[1].set_title('Distribution by Source and Class', fontsize=14, fontweight='bold')
        axes[1].legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(axis='y', alpha=0.3)
    else:
        # If no source column, hide second subplot
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nDistribution Statistics:")
    print(f"  - Total instances: {len(df)}")
    print(f"  - Number of classes: {len(class_counts)}")
    print(f"  - Distribution:")
    for label, count in class_counts.items():
        print(f"    {label}: {count} ({count/len(df)*100:.2f}%)")


def plot_correlations(df: pd.DataFrame,
                     feature_cols: Optional[List[str]] = None,
                     feature_columns: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (15, 12),
                     sample_size: Optional[int] = None) -> None:
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
        sample_size (Optional[int]): If specified, samples this many features (selected by variance).
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
        feature_cols = [col for col in feature_cols 
                       if col not in ['label', 'label_encoded', 'label_binary', 'source', 'label_original']]
    
    # Sample features if requested (for performance and clarity)
    if sample_size and len(feature_cols) > sample_size:
        # Select features with highest variance (most informative)
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = variances.head(sample_size).index.tolist()
        print(f"⚠️  Using {sample_size} features (highest variance) out of {len(df.columns)} for faster computation")
    
    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Print correlation statistics
    # Extract upper triangle (excluding diagonal) for statistics
    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    print(f"\nCorrelation Statistics:")
    print(f"  - Number of features: {len(feature_cols)}")
    print(f"  - Mean correlation: {upper_triangle.mean():.3f}")
    print(f"  - Max correlation: {upper_triangle.max():.3f}")
    print(f"  - Min correlation: {upper_triangle.min():.3f}")


def plot_time_series(df: pd.DataFrame,
                    instance_idx: int = 0,
                    sensors: List[str] = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'],
                    samples_per_sensor: int = 15,
                    figsize: Tuple[int, int] = (15, 10)) -> None:
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
    if instance_idx >= len(df):
        print(f"⚠️  Index {instance_idx} out of bounds (max: {len(df)-1})")
        return
    
    # Extract features for this instance (exclude label, source, etc.)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    instance_data = df.iloc[instance_idx][feature_cols].values
    
    # Reshape data: 15 samples × 6 sensors
    if len(instance_data) == samples_per_sensor * len(sensors):
        data_reshaped = instance_data.reshape(samples_per_sensor, len(sensors))
        time_steps = np.arange(1, samples_per_sensor + 1)
        
        # Create subplots for each sensor
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, sensor in enumerate(sensors):
            axes[i].plot(time_steps, data_reshaped[:, i], marker='o', linewidth=2, markersize=4)
            axes[i].set_xlabel('Time Sample', fontsize=10)
            axes[i].set_ylabel(f'{sensor}', fontsize=10)
            axes[i].set_title(f'Sensor {sensor}', fontsize=12, fontweight='bold')
            axes[i].grid(alpha=0.3)
        
        plt.suptitle(f'Time-Series Sensor Readings - Instance {instance_idx}', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Display instance label and source
        if 'label' in df.columns:
            label = df.iloc[instance_idx]['label']
            source = df.iloc[instance_idx].get('source', 'Unknown')
            print(f"\nInstance {instance_idx}:")
            print(f"  Label: {label}")
            print(f"  Source: {source}")
    else:
        print(f"⚠️  Data format not recognized. Expected: {samples_per_sensor * len(sensors)} features, "
              f"got: {len(instance_data)}")


def create_statistical_features(df: pd.DataFrame,
                               samples_per_sensor: int = 15,
                               num_sensors: int = 6) -> pd.DataFrame:
    """
    Create statistical features from time-series sensor data.
    
    This function transforms raw time-series data (90 features) into statistical features
    (48 features) by computing statistics for each sensor over the 15 time samples.
    This dimensionality reduction captures temporal patterns while reducing complexity.
    
    Statistical features computed for each sensor:
    - Mean: Central tendency
    - Std: Variability
    - Min/Max: Extreme values
    - Range: Amplitude (Max - Min)
    - Skewness: Distribution asymmetry
    - Kurtosis: Distribution tail heaviness
    - Trend: Linear trend slope
    
    Reference: Lesson 2 - Feature Engineering and Dimensionality Reduction
    
    Args:
        df (pd.DataFrame): DataFrame with original features (90 features = 15 samples × 6 sensors).
        samples_per_sensor (int): Number of time samples per sensor. Default is 15.
        num_sensors (int): Number of sensors. Default is 6.
        
    Returns:
        pd.DataFrame: DataFrame with statistical features (48 features) plus label and source columns.
        
    Example:
        >>> df_stats = create_statistical_features(df)
        >>> print(df_stats.shape)
        (463, 50)  # 48 features + label + source
    """
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    if len(feature_cols) != samples_per_sensor * num_sensors:
        print(f"⚠️  Format not recognized. Using all numeric features.")
        feature_cols = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]
        feature_cols = [col for col in feature_cols 
                       if col not in ['label', 'label_encoded', 'label_binary', 'source', 'label_original']]
    
    sensor_names = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    statistical_features = []
    
    print("Creating statistical features...")
    
    for idx in range(len(df)):
        instance_features = df.iloc[idx][feature_cols].values
        
        # Convert to numpy array with explicit float64 type
        # This avoids 'object' dtype issues with scipy.stats functions
        instance_features = np.array(instance_features, dtype=np.float64)
        
        # Reshape: 15 samples × 6 sensors
        if len(instance_features) == samples_per_sensor * num_sensors:
            data_reshaped = instance_features.reshape(samples_per_sensor, num_sensors)
            
            stats = {}
            for i, sensor in enumerate(sensor_names):
                sensor_data = data_reshaped[:, i]
                
                # Ensure sensor_data is a 1D float64 array
                sensor_data = np.array(sensor_data, dtype=np.float64).flatten()
                
                # Basic statistical features
                stats[f'{sensor}_mean'] = np.mean(sensor_data)
                stats[f'{sensor}_std'] = np.std(sensor_data)
                stats[f'{sensor}_min'] = np.min(sensor_data)
                stats[f'{sensor}_max'] = np.max(sensor_data)
                stats[f'{sensor}_range'] = np.max(sensor_data) - np.min(sensor_data)
                
                # Skewness and kurtosis (distribution shape)
                # Handle edge cases where std=0 or all values are identical
                if stats[f'{sensor}_std'] == 0 or len(np.unique(sensor_data)) == 1:
                    # If std=0 or all values identical, skewness and kurtosis = 0
                    stats[f'{sensor}_skew'] = 0.0
                    stats[f'{sensor}_kurtosis'] = 0.0
                else:
                    # Calculate skewness and kurtosis normally
                    skew_val = scipy_stats.skew(sensor_data)
                    kurt_val = scipy_stats.kurtosis(sensor_data)
                    # Replace NaN with 0 if necessary (safety check)
                    stats[f'{sensor}_skew'] = 0.0 if np.isnan(skew_val) else skew_val
                    stats[f'{sensor}_kurtosis'] = 0.0 if np.isnan(kurt_val) else kurt_val
                
                # Temporal features (linear trend)
                if len(sensor_data) > 1:
                    try:
                        trend = np.polyfit(range(len(sensor_data)), sensor_data, 1)[0]
                        stats[f'{sensor}_trend'] = 0.0 if np.isnan(trend) else trend
                    except (np.linalg.LinAlgError, ValueError):
                        # If trend calculation fails, set to 0
                        stats[f'{sensor}_trend'] = 0.0
            
            statistical_features.append(stats)
        else:
            # Fallback: compute statistics over all features
            instance_features = np.array(instance_features, dtype=np.float64)
            stats = {
                'mean': np.mean(instance_features),
                'std': np.std(instance_features),
                'min': np.min(instance_features),
                'max': np.max(instance_features),
                'range': np.max(instance_features) - np.min(instance_features)
            }
            statistical_features.append(stats)
    
    # Create DataFrame with statistical features
    stats_df = pd.DataFrame(statistical_features)
    
    # Add non-feature columns (label, source, etc.)
    for col in ['label', 'label_encoded', 'label_binary', 'source', 'label_original']:
        if col in df.columns:
            stats_df[col] = df[col].values
    
    print(f"✅ Statistical features created: {len(stats_df.columns)} columns")
    print(f"   Original features: {len(feature_cols)}")
    print(f"   Statistical features: {len(stats_df.columns) - len([c for c in stats_df.columns if c in ['label', 'label_encoded', 'label_binary', 'source', 'label_original']])}")
    
    return stats_df


def plot_pca(X: np.ndarray,
             y: Optional[np.ndarray] = None,
             n_components: int = 2,
             title: str = "PCA Visualization",
             figsize: Tuple[int, int] = (12, 8)) -> PCA:
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
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    elif n_components == 3:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            if y is not None:
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                                   c=y, cmap='viridis', alpha=0.6, s=50)
                plt.colorbar(scatter, label='Class')
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.6, s=50)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=10)
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)', fontsize=10)
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("⚠️  mpl_toolkits.mplot3d not available. 3D visualization skipped.")
    
    # Print explained variance
    print(f"\nExplained Variance:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")
    print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    return pca


def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_pred_proba: Optional[np.ndarray] = None,
                   model_name: str = "Model",
                   show_confusion_matrix: bool = True,
                   show_roc: bool = True,
                   show_report: bool = True) -> Dict:
    """
    Evaluate a classification model and display comprehensive metrics.
    
    This function computes and displays:
    - Accuracy, Precision, Recall, F1-score
    - Confusion matrix (visualization)
    - ROC curve and AUC (for binary classification)
    - Classification report (per-class metrics)
    
    Reference: Lesson 3 - Model Evaluation and Performance Metrics
    
    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        y_pred_proba (Optional[np.ndarray]): Predicted class probabilities (for ROC curve).
                                           Shape should be (n_samples,) for binary classification.
                                           Default is None.
        model_name (str): Model name for display. Default is "Model".
        show_confusion_matrix (bool): If True, displays confusion matrix. Default is True.
        show_roc (bool): If True, displays ROC curve (requires y_pred_proba). Default is True.
        show_report (bool): If True, displays classification report. Default is True.
        
    Returns:
        Dict: Dictionary containing evaluation metrics:
            - 'accuracy': float
            - 'precision': float
            - 'recall': float
            - 'f1_score': float
            - 'auc': float (if y_pred_proba provided and binary classification)
            - 'model_name': str
            
    Example:
        >>> results = evaluate_model(y_test, y_pred, y_pred_proba, model_name='Random Forest')
        >>> print(results['f1_score'])
        0.95
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Confusion matrix
    if show_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(np.unique(y_true)),
                   yticklabels=sorted(np.unique(y_true)),
                   cbar_kws={'label': 'Number of Instances'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predictions', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    # ROC curve (only for binary classification)
    if show_roc and y_pred_proba is not None:
        if len(np.unique(y_true)) == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print(f"AUC: {roc_auc:.4f}")
        else:
            print("⚠️  ROC curve available only for binary classification")
    
    # Classification report
    if show_report:
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model_name': model_name
    }
    
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        results['auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return results


def plot_feature_importances(model,
                            feature_names: List[str],
                            top_n: int = 20,
                            title: str = "Feature Importances",
                            figsize: Tuple[int, int] = (12, 8)) -> None:
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
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest, Gradient Boosting)
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models (Logistic Regression)
        # Take absolute value of coefficients for importance
        importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
    else:
        print("⚠️  Model does not have 'feature_importances_' or 'coef_' attribute")
        return
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Visualization
    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df['importance'], 
            color='steelblue', edgecolor='black', alpha=0.7)
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop {top_n} Most Important Features:")
    for i, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
