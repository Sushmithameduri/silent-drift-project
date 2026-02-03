"""
Silent Drift Detection Pipeline - Main Orchestrator

Coordinates the complete end-to-end workflow for detecting silent drift in ML
system telemetry using Mahalanobis distance-based statistical anomaly detection.

The pipeline detects gradual, subtle degradation in model performance metrics
that occurs due to data distribution shift, concept drift, or system degradation—
anomalies that are difficult to detect with simple threshold-based alerting.

Workflow (6 phases):
    1. Generate synthetic telemetry with injected silent drift (t=600+)
    2. Extract rolling statistical features (mean, std over 50-point windows)
    3. Split data: baseline 30% (clean, train) + monitoring 70% (test period)
    4. Fit Mahalanobis distance detector on baseline to learn normal distribution
    5. Score monitoring period; compute evaluation metrics with adaptive threshold
    6. Visualize results: plot anomaly scores with threshold overlay

Design Decisions:
    - 30% baseline split: Provides stable covariance matrix (rule: size >> features).
      Smaller baselines → unstable covariance → poor detector generalization.
    - Rolling features: Smooth noise while preserving temporal trends.
      Point metrics are noisy; distributions are interpretable.
    - Mahalanobis distance: Accounts for feature correlations and variance scaling.
      Superior to Euclidean distance for multivariate anomaly detection.
    - 95th percentile threshold: Adaptive to data; avoids hard-coded tuning.
      Can be replaced with ROC analysis or domain expert guidance.

Output Artifacts:
    - data/raw/telemetry.csv: Raw synthetic telemetry (1000 timesteps, 4 features)
    - data/processed/features.csv: Rolling features (mean/std, 950 rows post-NaN drop)
    - data/processed/anomaly_scores.png: Visualization with threshold overlay

See Also:
    - REFACTORING_SUMMARY.md: Code documentation standards and practices
    - src/anomaly_model.py: DriftDetector algorithm and implementation details
"""

from src.data_generator import generate_data
from src.preprocessing import rolling_features
from src.anomaly_model import DriftDetector
from src.evaluator import evaluate_anomalies
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output directories exist before saving artifacts.
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# =============================================================================
# Phase 1: Data Generation
# =============================================================================
# Generate 1000 timesteps of synthetic telemetry with silent drift.
# Drift begins at t=600 (60% into timeline), providing 600 baseline + 400 drift periods.
# Features: latency (control), confidence (drifts), embedding_norm (drifts).
df = generate_data()
df.to_csv("data/raw/telemetry.csv", index=False)

# =============================================================================
# Phase 2: Feature Engineering
# =============================================================================
# Transform raw metrics into rolling statistical features (mean, std).
# Rolling window (50 timesteps) smooths noise while preserving distribution changes.
# NaN rows from window startup are automatically dropped (reduces 1000 → 950 rows).
features = rolling_features(df)
features.to_csv("data/processed/features.csv", index=False)

# =============================================================================
# Phase 3: Train-Test Split
# =============================================================================
# Allocate 30% for baseline (training) and 70% for monitoring (test period).
# Baseline must contain ONLY clean, pre-drift data to learn accurate distribution.
# Rationale: Mixing drift into baseline violates stationarity assumption.
baseline_ratio = 0.3
split_idx = int(len(features) * baseline_ratio)
baseline = features.iloc[:split_idx]  # Clean baseline: learn normal distribution
monitor = features.iloc[split_idx:]   # Monitoring: score for drift detection

# =============================================================================
# Phase 4: Model Training and Scoring
# =============================================================================
# Fit Mahalanobis distance detector on baseline to learn distribution parameters.
# detector.fit() computes mean vector and inverse covariance matrix.
# Assumption: baseline is multivariate normal (reasonable for aggregated telemetry).
detector = DriftDetector()
detector.fit(baseline.values)

# Score monitoring period using learned baseline distribution.
# Higher scores indicate greater Mahalanobis distance from baseline (anomalous).
scores = detector.score(monitor.values)

# Print score statistics for threshold tuning and sanity checks.
# Baseline scores should be small (1-10 range for well-separated drift).
# Monitoring scores should show clear separation if drift is present.
print("Score Statistics (Monitoring Period):")
print(f"  Min: {scores.min():.2f}")
print(f"  Mean: {scores.mean():.2f}")
print(f"  95th percentile: {np.percentile(scores, 95):.2f}")
print(f"  Max: {scores.max():.2f}")

# =============================================================================
# Phase 5: Evaluation with Adaptive Threshold
# =============================================================================
# Determine threshold dynamically: 95th percentile of BASELINE scores.
# 
# CRITICAL: Use baseline distribution for threshold, NOT monitoring scores.
# Rationale: Baseline is clean (pre-drift); represents true "normal" variation.
# If threshold is derived from contaminated monitoring data, it shifts upward,
# causing high false negatives (misses most drift). This is an anti-pattern.
# 
# Industry best practice: Learn threshold from clean reference period only.
# Expected result: ~50-90% anomaly detection rate (realistic for injected drift).
#
# Alternative approaches: ROC analysis (if labels available), percentile sweep.
baseline_scores = detector.score(baseline.values)
threshold = np.percentile(baseline_scores, 95)

# Compute evaluation metrics: counts and ratios of anomalies detected.
results = evaluate_anomalies(scores, threshold)
print("\nEvaluation Results:")
print(f"  Total Points: {results['total_points']}")
print(f"  Anomalies Detected: {results['anomalies_detected']}")
print(f"  Anomaly Ratio: {results['anomaly_ratio']:.2%}")

# =============================================================================
# Phase 6: Visualization and Artifact Saving
# =============================================================================
# Create comprehensive visualization of anomaly scores over time.
# Plot shows: (1) score trajectory, (2) threshold line, (3) grid for readability.
# Saved to file for automated/headless execution (plt.show() would block).
plt.figure(figsize=(12, 6))

# Plot anomaly scores with legend.
plt.plot(scores, linewidth=1.5, label="Mahalanobis Distance Score", color="blue")

# Overlay threshold line (95th percentile of scores).
# Points above this line are classified as anomalies.
plt.axhline(y=threshold, linestyle="--", color="red", linewidth=2, 
            label=f"Detection Threshold (95th %ile: {threshold:.1f})")

# Labels and formatting for clarity.
plt.title("Anomaly Scores Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Time (monitoring period timesteps)", fontsize=12)
plt.ylabel("Mahalanobis Distance Score", fontsize=12)
plt.legend(fontsize=10, loc="upper left")
plt.grid(True, alpha=0.3, linestyle=":")

# Save to file.
plt.tight_layout()
plt.savefig("data/processed/anomaly_scores.png", dpi=100)
print(f"\nPlot saved to data/processed/anomaly_scores.png")
