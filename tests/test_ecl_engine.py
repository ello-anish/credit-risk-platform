"""ECL engine unit tests — Stage 3 > Stage 2 > Stage 1 for the same loan."""

from __future__ import annotations

import numpy as np

from ecl.engine import ECLInput, compute_ecl


def _base_input(n: int = 3, pd_12m=(0.01, 0.08, 0.15), dpd=(0, 0, 0),
                default_flag=(0, 0, 0)) -> ECLInput:
    return ECLInput(
        loan_id=np.array([1001, 1002, 1003][:n]),
        as_of_date=np.array(["2019-01-01"] * n),
        pd_12m=np.array(pd_12m[:n], dtype=float),
        pd_origination=np.array([0.05, 0.05, 0.05][:n], dtype=float),
        lgd=np.array([0.6, 0.6, 0.6][:n], dtype=float),
        ead=np.array([10000.0] * n, dtype=float),
        effective_rate=np.array([0.12] * n, dtype=float),
        remaining_months=np.array([24] * n, dtype=int),
        days_past_due=np.array(dpd[:n], dtype=int),
        default_flag_at_asof=np.array(default_flag[:n], dtype=int),
    )


def test_stage_assignment_order():
    """A low-PD loan -> Stage 1, SICR triggered loan -> Stage 2, defaulted -> Stage 3."""
    # Loan 1: PD unchanged from origination, no DPD -> Stage 1
    # Loan 2: PD doubled, no DPD -> Stage 2
    # Loan 3: defaulted -> Stage 3
    inp = _base_input(
        pd_12m=(0.05, 0.15, 0.90),   # loan 2 tripled, loan 3 almost 1
        dpd=(0, 0, 120),
        default_flag=(0, 0, 1),
    )
    # Adjust pd_origination so loan 1 is clearly Stage 1
    inp.pd_origination = np.array([0.05, 0.05, 0.05])

    out = compute_ecl(inp)
    assert out["stage"].tolist() == [1, 2, 3], out["stage"].tolist()


def test_stage3_ecl_exceeds_stage2_exceeds_stage1_same_loan():
    """For comparable loans, Stage 3 ECL > Stage 2 ECL > Stage 1 ECL.

    Stage 2 uses lifetime ECL at the CURRENT PD; Stage 3 reflects that the loan
    has already defaulted, so PD_current ~= 1.0 — this is what drives the
    ECL ordering in practice. If we forced the same PD across stages, Stage 2
    and Stage 3 lifetime ECL would be identical (LGD floor aside), which is
    not a meaningful test of the engine.
    """
    common = dict(
        loan_id=np.array([1]),
        as_of_date=np.array(["2019-01-01"]),
        lgd=np.array([0.6], dtype=float),
        ead=np.array([10000.0], dtype=float),
        effective_rate=np.array([0.12], dtype=float),
        remaining_months=np.array([36], dtype=int),
    )

    # Stage 1: stable PD, no DPD, no default.
    s1 = ECLInput(
        **common,
        pd_12m=np.array([0.04], dtype=float),
        pd_origination=np.array([0.04], dtype=float),
        days_past_due=np.array([0], dtype=int),
        default_flag_at_asof=np.array([0], dtype=int),
    )

    # Stage 2: current PD has doubled (SICR), no DPD, no default.
    s2 = ECLInput(
        **common,
        pd_12m=np.array([0.20], dtype=float),
        pd_origination=np.array([0.04], dtype=float),
        days_past_due=np.array([0], dtype=int),
        default_flag_at_asof=np.array([0], dtype=int),
    )

    # Stage 3: loan has defaulted — PD ~= 1.0, 120+ DPD.
    s3 = ECLInput(
        **common,
        pd_12m=np.array([0.99], dtype=float),
        pd_origination=np.array([0.04], dtype=float),
        days_past_due=np.array([120], dtype=int),
        default_flag_at_asof=np.array([1], dtype=int),
    )

    e1 = compute_ecl(s1).iloc[0]
    e2 = compute_ecl(s2).iloc[0]
    e3 = compute_ecl(s3).iloc[0]

    assert e1["stage"] == 1
    assert e2["stage"] == 2
    assert e3["stage"] == 3
    assert e1["reported_ecl"] < e2["reported_ecl"], (
        f"Stage1 {e1['reported_ecl']:,.2f} >= Stage2 {e2['reported_ecl']:,.2f}"
    )
    assert e2["reported_ecl"] < e3["reported_ecl"], (
        f"Stage2 {e2['reported_ecl']:,.2f} >= Stage3 {e3['reported_ecl']:,.2f}"
    )


def test_stage3_lgd_floor_applied():
    """Stage 3 should bump an LGD below the floor up to the floor value."""
    from pipeline.config import CFG
    floor = float(CFG["ecl"]["lgd_floor_stage3"])

    inp = ECLInput(
        loan_id=np.array([1]),
        as_of_date=np.array(["2019-01-01"]),
        pd_12m=np.array([0.5], dtype=float),
        pd_origination=np.array([0.01], dtype=float),
        lgd=np.array([0.001], dtype=float),         # absurdly low
        ead=np.array([10000.0], dtype=float),
        effective_rate=np.array([0.12], dtype=float),
        remaining_months=np.array([12], dtype=int),
        days_past_due=np.array([120], dtype=int),
        default_flag_at_asof=np.array([1], dtype=int),
    )
    out = compute_ecl(inp).iloc[0]
    assert out["stage"] == 3
    assert out["lgd"] >= floor - 1e-9, f"lgd {out['lgd']} < floor {floor}"
