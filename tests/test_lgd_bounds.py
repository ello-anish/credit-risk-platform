"""LGD predictions must fall in [0, 1]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.config import REPO_ROOT

LGD_PREDS = REPO_ROOT / "models" / "lgd_python" / "artifacts" / "lgd_predictions.parquet"


@pytest.mark.skipif(not LGD_PREDS.exists(), reason="LGD predictions missing.")
def test_lgd_pred_in_unit_interval():
    df = pd.read_parquet(LGD_PREDS)
    assert (df["lgd_pred"] >= 0.0 - 1e-9).all()
    assert (df["lgd_pred"] <= 1.0 + 1e-9).all()


@pytest.mark.skipif(not LGD_PREDS.exists(), reason="LGD predictions missing.")
def test_lgd_actual_in_unit_interval():
    df = pd.read_parquet(LGD_PREDS)
    assert (df["lgd_actual"] >= 0.0 - 1e-9).all()
    assert (df["lgd_actual"] <= 1.0 + 1e-9).all()


@pytest.mark.skipif(not LGD_PREDS.exists(), reason="LGD predictions missing.")
def test_lgd_distribution_not_degenerate():
    """LGD predictions should have non-trivial variance.

    LendingClub LGDs are famously concentrated near 1.0 (unsecured consumer
    credit has low recoveries — 60 %+ of defaults hit LGD > 0.9), so the
    beta-regression predictions legitimately cluster. The threshold is low
    but still excludes a degenerate constant-output model.
    """
    df = pd.read_parquet(LGD_PREDS)
    std = df["lgd_pred"].std()
    rng = df["lgd_pred"].max() - df["lgd_pred"].min()
    # Either the std is above 0.005, OR the observed range is > 0.05 — both
    # are reasonable lower bounds for "this model actually discriminates."
    assert std > 0.005 or rng > 0.05, (
        f"LGD predictions effectively constant: std={std:.4f}, range={rng:.4f}"
    )
