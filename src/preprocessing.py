"""
Feature Engineering: Rolling Statistical Feature Extraction

Transforms raw telemetry into time-windowed aggregated features suitable for
statistical anomaly detection.

Rationale:
    - Rolling statistics smooth measurement noise while preserving temporal trends.
    - Mean captures central tendency; std captures dispersion (variability).
    - Together, they enable the Mahalanobis detector to identify when the joint
      distribution deviates from baseline (indicating drift or anomalies).
    - This representation converts point anomalies into distributional anomalies,
      improving detection of gradual, subtle drift patterns.

Design notes:
    - Window size (default 50) is a hyperparameter; tune based on data frequency
      and expected drift speed. Smaller windows = more responsive but noisier.
    - First (window-1) rows become NaN due to rolling initialization; these are dropped.
    - Assumes input is temporally ordered; shuffled data produces meaningless features.
    - Suitable for stationary or slowly drifting baselines; not for highly seasonal data.
"""

from typing import Optional
import pandas as pd


def rolling_features(
    df: pd.DataFrame,
    window: int = 50
) -> pd.DataFrame:
    """
    Extract rolling statistical features from raw telemetry.

    Computes rolling mean and standard deviation for each raw metric over a
    fixed sliding window. This transforms point metrics into distributional
    features, enabling statistical anomaly detection.

    Args:
        df: Raw telemetry DataFrame. Expected columns:
            ["time", "latency", "confidence", "embedding_norm"].
            Must be in chronological order (index represents time).
        window: Rolling window size in timesteps. Default 50 balances noise
               smoothing against responsiveness to rapid changes. Adjust based on:
               - Data frequency (higher frequency → larger window for stable stats)
               - Expected drift speed (faster drift → smaller window)
               - Computational budget (larger window = fewer output rows)

    Returns:
        pd.DataFrame: Shape (n_points - window + 1, 6) containing:
            - latency_mean, confidence_mean, embedding_mean: rolling averages
            - latency_std, confidence_std, embedding_std: rolling standard deviations
        First (window-1) rows are discarded due to NaN from rolling window startup.

    Note:
        - Assumption: Input is temporally ordered. Shuffled data yields invalid features.
        - Limitation: Data loss of (window-1) rows is inherent to rolling window design.
          Plan baseline size accordingly (e.g., if window=50, need ≥300 baseline rows
          for stable covariance estimation).
        - All NaN values from rolling window are dropped; no interpolation attempted.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> raw = pd.DataFrame({
        ...     "time": range(100),
        ...     "latency": np.random.normal(100, 10, 100),
        ...     "confidence": np.random.normal(0.9, 0.02, 100),
        ...     "embedding_norm": np.random.normal(10, 0.5, 100)
        ... })
        >>> features = rolling_features(raw, window=30)
        >>> print(features.shape)
        (71, 6)  # 100 - 30 + 1 = 71 rows after dropping NaNs
    """
    features = pd.DataFrame()

    # Compute rolling mean: smooths noise, captures trend in central tendency.
    features["latency_mean"] = df["latency"].rolling(window).mean()
    features["confidence_mean"] = df["confidence"].rolling(window).mean()
    features["embedding_mean"] = df["embedding_norm"].rolling(window).mean()

    # Compute rolling std: captures variability/dispersion.
    # Elevated std can indicate system instability or increasing variance (precursor to drift).
    features["latency_std"] = df["latency"].rolling(window).std()
    features["confidence_std"] = df["confidence"].rolling(window).std()
    features["embedding_std"] = df["embedding_norm"].rolling(window).std()

    # Drop NaN rows from rolling window initialization (first window-1 rows).
    # These rows have incomplete windows and invalid statistics.
    features = features.dropna()
    return features
