"""Standalone isotonic-calibration helper.

The main train.py already does calibration inline via ``CalibratedClassifierCV``.
This module exposes the primitive for re-calibration against a fresh slice
(e.g. a PIT snapshot) without retraining the underlying base estimator.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def recalibrate_isotonic(
    base_probs: np.ndarray,
    y_true: np.ndarray,
) -> IsotonicRegression:
    """Fit an isotonic mapping of base probabilities to observed default rates."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(base_probs, y_true)
    return iso
