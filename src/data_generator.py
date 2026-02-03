"""
Synthetic Telemetry Data Generator for Silent Drift Detection

Generates synthetic ML system telemetry with injected silent drift.

Silent drift is a gradual, hard-to-detect degradation in model metrics that
occurs due to data distribution shift, concept drift, or system degradation.
This module simulates realistic scenarios where performance decays slowly over time,
challenging detectors with subtle anomalies.

Features generated:
    - latency: API response time (ms). Normal: μ=100, σ=10. Remains stable (control).
    - confidence: Model prediction confidence. Normal: μ=0.9, σ=0.02. Drifts downward.
    - embedding_norm: Embedding magnitude. Normal: μ=10, σ=0.5. Drifts downward.

Drift mechanism (post-drift_start):
    - Confidence decreases by 0.0003 per timestep (cumulative linear decay).
    - Embedding_norm decreases by 0.002 per timestep (cumulative linear decay).
    - Latency remains normal (acts as negative control for specificity testing).
"""

from typing import Tuple
import numpy as np
import pandas as pd


def generate_data(n_points: int = 1000, drift_start: int = 600) -> pd.DataFrame:
    """
    Generate synthetic telemetry data with silent drift.

    Creates a realistic dataset where model metrics gradually degrade after a
    specific temporal point, simulating real-world performance degradation.

    Args:
        n_points: Total number of timesteps to simulate. Default 1000 provides
                 600 baseline points and 400 monitoring points (ideal ratio ~60/40).
        drift_start: Timestep at which silent drift begins (inclusive).
                    Recommendation: 60-70% of n_points to allow sufficient baseline.
                    In this case, 600/1000 = 60%.

    Returns:
        pd.DataFrame: Shape (n_points, 4) with columns:
            - time: Timestep index (0 to n_points-1)
            - latency: Request latency in ms
            - confidence: Model confidence [0-1]
            - embedding_norm: Embedding magnitude (normalized)

    Note:
        - Randomness is unbounded; externally seed RNG for reproducibility.
        - Drift is deterministic and cumulative (linear per-timestep degradation).
        - Latency serves as negative control: should not trigger drift detector
          (validates specificity on stable metrics).

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = generate_data(n_points=100, drift_start=60)
        >>> print(df.shape)
        (100, 4)
        >>> print(df[50:55])  # Baseline period (before drift)
    """
    data = []

    for t in range(n_points):
        # Baseline telemetry: normally distributed around typical production values.
        latency = np.random.normal(100, 10)         # Stable throughout (negative control)
        confidence = np.random.normal(0.9, 0.02)    # Subject to drift
        embedding_norm = np.random.normal(10, 0.5)  # Subject to drift

        # Inject cumulative silent drift: linear decay after drift_start.
        # Rationale: Linear decay models realistic degradation (e.g., cache hit rate decay,
        # token budget saturation, model staleness accumulating over time).
        if t > drift_start:
            confidence -= (t - drift_start) * 0.0003
            embedding_norm -= (t - drift_start) * 0.002

        data.append([t, latency, confidence, embedding_norm])

    df = pd.DataFrame(
        data,
        columns=["time", "latency", "confidence", "embedding_norm"]
    )
    return df
