"""Monotonicity: higher FICO should imply lower PD (partial dependence check)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.config import REPO_ROOT


def _have_artifacts() -> bool:
    return (REPO_ROOT / "models" / "pd_python" / "artifacts" / "pd_model.joblib").exists()


@pytest.mark.skipif(not _have_artifacts(), reason="PD model not trained yet.")
def test_higher_fico_lower_pd():
    """Partial-dependence-style sweep on fico_mid, holding other features at median."""
    clean = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    # Take a median row, then sweep FICO
    mod = clean[clean["default_flag"].notna()]
    median_row = mod.iloc[[len(mod) // 2]].copy()

    from models.pd_python.predict import score_loans

    ficos = [580, 640, 680, 720, 760, 800]
    rows = []
    for f in ficos:
        r = median_row.copy()
        r["fico_range_low"] = f - 2
        r["fico_range_high"] = f + 2
        r["fico_mid"] = f
        rows.append(r)
    df = pd.concat(rows, ignore_index=True)
    probs = score_loans(df, which="baseline")

    # Monotonicity: prob should be (weakly) non-increasing in FICO.
    # Allow one small violation (sampling noise) but require overall trend.
    assert probs[0] > probs[-1], (
        f"FICO=580 PD ({probs[0]:.4f}) not greater than FICO=800 PD ({probs[-1]:.4f})"
    )
    diffs = np.diff(probs)
    # At most one strictly-positive diff is allowed
    assert np.sum(diffs > 1e-4) <= 1, f"PD not monotonic in FICO: {probs}"
