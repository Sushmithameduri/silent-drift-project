"""
Evaluation Utilities: Anomaly Detection Result Summary

Computes interpretable summary statistics for anomaly detection results.

Design philosophy:
    - Keep metrics simple (counts, ratios) for transparency and interpretability.
    - Threshold selection is left to caller (data-driven, not hardcoded).
    - No automatic thresholding; caller has full control over decision boundary.
    - In production with ground truth labels, consider precision, recall, ROC curves,
      and F1-score. In unsupervised drift detection (typical use), these metrics
      require synthetic labels and are less meaningful.

Typical workflow:
    1. Score data: scores = detector.score(X)
    2. Choose threshold (e.g., 95th percentile of baseline scores)
    3. Evaluate: results = evaluate_anomalies(scores, threshold)
    4. Inspect results for alert frequency and detection rate
    5. Adjust threshold based on alert cost/tolerance (iterate)
"""

from typing import Dict
import numpy as np


def evaluate_anomalies(
    scores: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Evaluate anomaly detection results for a given threshold.

    Classifies samples as anomalies if their anomaly score exceeds the threshold,
    then computes summary statistics (counts and ratios).

    Args:
        scores: Array of anomaly scores from detector, shape (n_samples,).
               Typically output from DriftDetector.score().
               Values should be non-negative (e.g., Mahalanobis distances).
        threshold: Decision boundary for anomaly classification.
                  Threshold selection is critical and application-dependent:
                  - Too low → false positives, alert fatigue
                  - Too high → missed detections (high false negatives)
                  Recommendation: Use percentile of baseline scores (e.g., 95th ≈ 5% FPR)
                  or perform ROC analysis if evaluation labels available.

    Returns:
        Dict with keys:
        - "total_points": Total number of samples evaluated (len(scores)).
        - "anomalies_detected": Count of samples with score > threshold.
        - "anomaly_ratio": Fraction of samples classified as anomalous, [0, 1].
                          Can be interpreted as alert rate for monitoring dashboards.

    Example:
        >>> scores = np.array([0.5, 15.0, 2.0, 25.0, 8.0])
        >>> results = evaluate_anomalies(scores, threshold=10)
        >>> print(results)
        {'total_points': 5, 'anomalies_detected': 2, 'anomaly_ratio': 0.4}
        # Two samples (15.0, 25.0) exceed threshold.

    Note:
        - Threshold should be determined on separate validation/baseline data
          to avoid overfitting to monitoring period.
        - For time-series data, consider using sliding-window percentiles
          for adaptive thresholds that account for seasonal patterns.
    """
    # Binary classification: score > threshold → anomaly (1), else normal (0).
    anomalies = scores > threshold

    return {
        "total_points": len(scores),
        "anomalies_detected": int(anomalies.sum()),
        "anomaly_ratio": float(anomalies.mean())
    }
