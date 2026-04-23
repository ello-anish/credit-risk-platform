"""Scoring-time PD Python predict — loads joblib artefacts and scores new loans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.config import REPO_ROOT
from pipeline.features import NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES

ART = REPO_ROOT / "models" / "pd_python" / "artifacts"


def load_artifacts() -> tuple[Any, Any, Any]:
    """Return (encoder, calibrated_baseline, gbm_challenger)."""
    enc = joblib.load(ART / "pd_encoder.joblib")
    baseline = joblib.load(ART / "pd_model.joblib")
    gbm = joblib.load(ART / "pd_gbm.joblib")
    return enc, baseline, gbm


def score_loans(
    features_df: pd.DataFrame,
    which: str = "baseline",
) -> np.ndarray:
    """Return PD_12m predictions for the given rows.

    Args:
        features_df: Must include all NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES columns.
        which: ``"baseline"`` (calibrated logit) or ``"gbm"``.
    """
    enc, baseline, gbm = load_artifacts()
    clf = {"baseline": baseline, "gbm": gbm}[which]
    Xraw = features_df[NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES]
    Xw = enc.transform(Xraw)
    # Filter to the columns the fitted model knows about
    if hasattr(clf, "feature_names_in_"):
        wanted = list(clf.feature_names_in_)
        for c in wanted:
            if c not in Xw.columns:
                Xw[c] = 0.0
        Xw = Xw[wanted]
    return clf.predict_proba(Xw)[:, 1]
