"""
Unit Tests: Drift Detector Integration Tests

Validates core functionality of the anomaly detection pipeline.

Test scope:
    - Smoke test: verifies detector executes without exceptions.
    - Shape validation: ensures output dimensions match expected contract.
    - Does NOT validate score correctness (requires synthetic truth labels).
    - Does NOT cover edge cases (singular matrices, NaN values, etc.).

To expand coverage:
    - Add fixtures with known baseline and synthetic anomalies.
    - Validate score ranges and mathematical invariants (e.g., scores >= 0).
    - Test pathological covariance matrices (singular, ill-conditioned).
    - Add integration tests with synthetic drift injection.

Run tests: pytest tests/test_anomaly.py -v
"""

import numpy as np
from src.anomaly_model import DriftDetector


def test_detector_runs() -> None:
    """
    Smoke test: verify detector basic end-to-end functionality.

    Ensures:
    - fit() accepts random baseline data without exceptions.
    - score() returns output with correct shape.
    - No runtime errors are raised during execution.

    This is a minimal sanity check only; does not validate detection quality
    or score correctness. A full test would require synthetic truth labels.

    Note:
        - Uses random normal data (N(0,1)) for simplicity.
        - Covariance is well-conditioned for random normal data (no singularity).
        - Scores on training data should be small (~1-10 range for 3D normal data)
          but this test only verifies shape, not value ranges.
    """
    # Create synthetic baseline: 100 samples, 3 features, standard normal.
    X = np.random.normal(0, 1, (100, 3))

    # Initialize and fit detector on baseline.
    detector = DriftDetector()
    detector.fit(X)

    # Score the same baseline data (for simplicity; ideally use separate test set).
    # Scores should be small since we're scoring on training data.
    scores = detector.score(X)

    # Verify output shape matches input (n_samples,).
    assert len(scores) == 100, (
        f"Expected 100 scores (one per sample), got {len(scores)}"
    )
    assert scores.ndim == 1, f"Expected 1D array, got shape {scores.shape}"
    # TODO: Add assertions for score ranges (e.g., all >= 0) tracked in issue #15.
    # TODO: Add test with known anomalies to validate detection quality (issue #16).
