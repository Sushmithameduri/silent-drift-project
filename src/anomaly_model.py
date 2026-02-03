"""
Anomaly Detection: Mahalanobis Distance-Based Drift Detector

Implements statistical anomaly detection using Mahalanobis distance, a metric
for measuring deviation from a multivariate normal distribution.

Why Mahalanobis Distance?
    - Accounts for correlations between features (unlike Euclidean distance).
    - Scales each feature by its variance (high-variance features don't dominate).
    - Assumes baseline data is approximately multivariate normal (reasonable for
      aggregated telemetry over sufficient time windows).
    - Computationally efficient and interpretable for production monitoring.
    - Well-established in industrial anomaly detection (e.g., Hotelling T² test).

Mathematical Foundation:
    score(x) = (x - μ)^T × Σ^-1 × (x - μ)
    where μ is baseline mean vector, Σ is covariance matrix.
    Higher scores indicate greater deviation from baseline distribution.

Limitations and Mitigations:
    - Requires n_samples >> n_features for stable covariance (rule: >10× features).
      Mitigation: Use sufficient baseline data (≥300-500 points for 5-10 features).
    - Covariance matrix must be invertible (non-singular).
      Mitigation: Add regularization (small constant to diagonal) if matrix is ill-conditioned.
    - Sensitive to outliers in baseline training data.
      Mitigation: Preprocess/validate baseline for known anomalies.
    - Assumes stationarity of baseline; violates assumptions if baseline is non-stationary.
      Mitigation: Validate baseline is drift-free before model training.
"""

from typing import Tuple, Optional
import numpy as np


class DriftDetector:
    """
    Mahalanobis distance-based anomaly detector for multivariate data.

    Learns the mean and covariance structure of a baseline (normal) distribution,
    then scores new samples by their Mahalanobis distance from that distribution.
    Samples with high scores are flagged as anomalies/drift.

    Attributes:
        mean (np.ndarray): Feature-wise mean of baseline training data. Shape: (n_features,).
                          Initialized to None; set by fit().
        cov_inv (np.ndarray): Inverse of baseline covariance matrix.
                             Shape: (n_features, n_features).
                             Initialized to None; set by fit().

    Example:
        >>> import numpy as np
        >>> detector = DriftDetector()
        >>> baseline = np.random.randn(100, 5)  # 100 samples, 5 features
        >>> detector.fit(baseline)
        >>> new_data = np.random.randn(10, 5)
        >>> scores = detector.score(new_data)
        >>> anomaly_mask = scores > 10  # Threshold tuning required
        >>> print(f\"Detected {anomaly_mask.sum()} anomalies\")
    """

    def __init__(self) -> None:
        """
        Initialize detector with unset parameters.

        Attributes (mean, cov_inv) are populated by fit() method.
        Calling score() before fit() raises RuntimeError.
        """
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        """
        Learn baseline distribution from training data.

        Computes mean and inverse covariance matrix that define the baseline
        (normal) distribution. Must be called before score().

        Args:
            X: Baseline feature matrix of shape (n_samples, n_features).
               Assumed to represent "normal" (non-drifted) behavior.
               Recommendation: n_samples >= 10 * n_features (more is better).
               Example: 300-500 samples for 5-10 features.

        Raises:
            np.linalg.LinAlgError: If covariance matrix is singular (non-invertible).
               Common causes: n_samples < n_features, duplicate rows, perfect multicollinearity.
               Mitigation: Increase baseline size or add L2 regularization to covariance diagonal.

        Side Effects:
            Sets self.mean and self.cov_inv attributes.

        Note:
            - Covariance computed with rowvar=False: each column is a feature.
            - No regularization applied by default; add if matrix singularity occurs.
        """
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)  # rowvar=False: columns are features
        self.cov_inv = np.linalg.inv(cov)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance scores for samples.

        Measures how far each sample lies from the learned baseline distribution.
        Higher scores indicate greater deviation (potential drift/anomaly).

        Threshold selection is application-specific:
            - Too low: false positives, alert fatigue
            - Too high: missed detections
        Common approaches:
            1. Percentile of baseline scores (e.g., 95th ≈ 5% false positive rate)
            2. ROC curve analysis (if labeled evaluation data available)
            3. Domain expert tuning based on alert cost/tolerance

        Args:
            X: Feature matrix of shape (n_samples, n_features) to score.
               Must have same feature dimensionality as training data.

        Returns:
            np.ndarray: Mahalanobis distances, shape (n_samples,).
            Each element is a non-negative scalar (mathematically guaranteed).
            Typical ranges:
                - Normal baseline data: scores in range 0-5
                - Moderate drift/anomalies: scores 5-20
                - Severe anomalies: scores 20+
            (Ranges depend on dimensionality and distribution overlap.)

        Raises:
            RuntimeError: If fit() has not been called (mean or cov_inv is None).

        Note:
            - Current implementation iterates over samples (slow for large X).
              Future optimization: vectorize with matrix operations for 10-100× speedup.
            - Scores are mathematical Mahalanobis distances; no normalization applied.

        Example:
            >>> scores = detector.score(new_data)
            >>> threshold = np.percentile(baseline_scores, 95)  # 95th percentile
            >>> anomalies = scores > threshold
        """
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError(
                "Detector not fitted. Call fit(X) with baseline data before score(X)."
            )

        scores = []
        for x in X:
            # Compute: (x - μ)^T × Σ^-1 × (x - μ)
            diff = x - self.mean
            score = diff.T @ self.cov_inv @ diff
            scores.append(score)
        return np.array(scores)
