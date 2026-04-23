"""Feature-engineering tests: no look-ahead, IV filter correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    LEAKAGE_COLUMNS,
    NUMERIC_CANDIDATES,
    CATEGORICAL_CANDIDATES,
    WoEEncoder,
    compute_woe_iv,
    select_features_by_iv,
)


# ---------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------
POST_ORIG_FIELDS = {
    "recoveries", "collection_recovery_fee", "total_pymnt", "total_pymnt_inv",
    "total_rec_int", "total_rec_late_fee", "total_rec_prncp", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "out_prncp", "out_prncp_inv",
}


def test_leakage_columns_are_banned():
    """All known post-origination fields must appear in LEAKAGE_COLUMNS."""
    banned = set(LEAKAGE_COLUMNS.keys())
    missing = POST_ORIG_FIELDS - banned
    assert not missing, f"Expected these to be banned: {missing}"


def test_candidate_features_disjoint_from_leakage():
    """No NUMERIC/CATEGORICAL candidate can also be a leakage column."""
    all_candidates = set(NUMERIC_CANDIDATES) | set(CATEGORICAL_CANDIDATES)
    overlap = all_candidates & set(LEAKAGE_COLUMNS.keys())
    assert not overlap, f"Candidates overlap with leakage list: {overlap}"


# ---------------------------------------------------------------------
# WoE / IV math
# ---------------------------------------------------------------------
def test_compute_woe_iv_known_case():
    """Hand-computed case: 50/50 split with 2x default rate in bin A."""
    x = pd.Series(["A"] * 50 + ["B"] * 50)
    y = pd.Series([1] * 20 + [0] * 30 + [1] * 10 + [0] * 40)
    woe, iv = compute_woe_iv(x, y)
    assert "A" in woe and "B" in woe
    # Bin A has higher default rate -> WoE_A should be NEGATIVE
    # (using WoE = log(%neg / %pos): more bads => lower %neg/%pos => negative WoE)
    assert woe["A"] < woe["B"], (woe)
    assert iv > 0


def test_iv_filter_kept_range():
    """select_features_by_iv only keeps features with iv_min <= iv <= iv_max."""
    enc = WoEEncoder(numeric=[], categorical=[])
    enc.iv_ = {
        "too_low": 0.005,
        "good_one": 0.10,
        "good_two": 0.25,
        "too_high": 0.90,
    }
    kept, iv_df, dropped = select_features_by_iv(enc)
    assert set(kept) == {"good_one", "good_two"}
    reasons = {d["feature"]: d["reason"] for d in dropped}
    assert "too_low" in reasons and "too_high" in reasons
