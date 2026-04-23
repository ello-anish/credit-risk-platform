"""IFRS 9 Expected Credit Loss engine.

Produces, per (loan_id, as_of_date, scenario):

    pd_12m, pd_lifetime, lgd, ead,
    ecl_12m       = PD_12m * LGD * EAD * DF
    ecl_lifetime  = sum_t ( marginal_PD_t * LGD * EAD_t * DF_t )
    stage         = 1 / 2 / 3 per SICR rules
    reported_ecl  = ecl_12m (if Stage 1) else ecl_lifetime

Staging rules (Tier 1, simple; overridable in config.yml:ecl):
    Stage 1: baseline — 12m ECL
    Stage 2: SICR trigger — 12m PD has more-than-doubled since origination
             OR days_past_due >= 30. Lifetime ECL.
    Stage 3: credit-impaired — default_flag=1 at as_of OR days_past_due >= 90.
             Lifetime ECL with LGD floor (config.yml:ecl:lgd_floor_stage3).

Lifetime PD (Tier 1):
    Geometric extrapolation from 12m PD:
        marginal_PD(t) = 1 - (1 - PD_12m)^(1/12) applied monthly
    Tier 2 replaces this with survival-derived marginal PDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.config import CFG
from pipeline.logging_utils import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------
def assign_stage(
    pd_current: np.ndarray,
    pd_origination: np.ndarray,
    days_past_due: np.ndarray,
    default_flag_at_asof: np.ndarray,
) -> np.ndarray:
    """Return an integer stage vector (1 / 2 / 3)."""
    cfg = CFG["ecl"]
    sicr_ratio = float(cfg["sicr_pd_ratio"])
    sicr_dpd = int(cfg["sicr_dpd_threshold"])
    default_dpd = int(cfg["default_dpd_threshold"])

    pd_current = np.asarray(pd_current, dtype=float)
    pd_origination = np.asarray(pd_origination, dtype=float)
    days_past_due = np.asarray(days_past_due, dtype=float)
    default_flag_at_asof = np.asarray(default_flag_at_asof, dtype=int)

    stage = np.full_like(pd_current, 1, dtype=int)

    # SICR checks (Stage 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pd_ratio = np.where(pd_origination > 0, pd_current / pd_origination, 1.0)
    sicr = (pd_ratio > sicr_ratio) | (days_past_due >= sicr_dpd)
    stage = np.where(sicr, 2, stage)

    # Stage 3: defaulted or 90+ DPD
    stage3 = (default_flag_at_asof == 1) | (days_past_due >= default_dpd)
    stage = np.where(stage3, 3, stage)

    return stage.astype(int)


# ---------------------------------------------------------------------
# Lifetime PD extrapolation
# ---------------------------------------------------------------------
def marginal_pd_schedule(pd_12m: float, remaining_months: int) -> np.ndarray:
    """Convert an annual 12m PD into a monthly marginal-PD vector.

    Geometric extrapolation:
        monthly_survival = (1 - pd_12m) ** (1/12)
        marginal_pd_t = monthly_survival^(t-1) * (1 - monthly_survival)
    """
    pd_12m = float(np.clip(pd_12m, 1e-6, 0.999999))
    monthly_survival = (1.0 - pd_12m) ** (1.0 / 12.0)
    t = np.arange(1, max(remaining_months, 1) + 1)
    # Marginal = surv^(t-1) * (1 - surv)
    return (monthly_survival ** (t - 1)) * (1.0 - monthly_survival)


def lifetime_pd(pd_12m: float, remaining_months: int) -> float:
    """Cumulative lifetime PD = 1 - (1 - pd_12m) ** (remaining_months/12)."""
    pd_12m = float(np.clip(pd_12m, 1e-6, 0.999999))
    years = remaining_months / 12.0
    return float(1.0 - (1.0 - pd_12m) ** years)


# ---------------------------------------------------------------------
# ECL computation
# ---------------------------------------------------------------------
@dataclass
class ECLInput:
    loan_id: np.ndarray
    as_of_date: np.ndarray                 # date
    pd_12m: np.ndarray                     # shape (n,)
    pd_origination: np.ndarray             # shape (n,)
    lgd: np.ndarray                        # shape (n,)
    ead: np.ndarray                        # shape (n,)  — outstanding principal
    effective_rate: np.ndarray             # annual, fraction
    remaining_months: np.ndarray           # int
    days_past_due: np.ndarray              # int
    default_flag_at_asof: np.ndarray       # 0/1


def compute_ecl(inp: ECLInput, scenario: str = "baseline",
                model_version: str = "v1") -> pd.DataFrame:
    """Vectorised ECL over a cohort of loans."""
    n = len(inp.loan_id)
    if n == 0:
        return pd.DataFrame()

    lgd_floor = float(CFG["ecl"]["lgd_floor_stage3"])

    stages = assign_stage(
        inp.pd_12m, inp.pd_origination, inp.days_past_due, inp.default_flag_at_asof
    )

    # Apply Stage 3 LGD floor
    lgd_eff = np.where(stages == 3,
                       np.maximum(inp.lgd, lgd_floor),
                       inp.lgd)

    # 12m ECL = PD_12m * LGD * EAD * DF (with 12m discount)
    df_12m = 1.0 / (1.0 + inp.effective_rate) ** 1.0  # single-year discount
    ecl_12m = inp.pd_12m * lgd_eff * inp.ead * df_12m

    # Lifetime ECL
    ecl_lifetime = np.zeros(n, dtype=float)
    pd_lifetime_vec = np.zeros(n, dtype=float)
    for i in range(n):
        rm = int(max(1, inp.remaining_months[i]))
        marginals = marginal_pd_schedule(inp.pd_12m[i], rm)
        # Simple EAD assumption: outstanding declines linearly to zero over rm months.
        # This is a per-month ladder.
        ead_schedule = inp.ead[i] * np.linspace(1.0, 0.0, rm, endpoint=False) \
            if rm > 1 else np.array([inp.ead[i]])
        if len(ead_schedule) < len(marginals):
            ead_schedule = np.pad(ead_schedule, (0, len(marginals) - len(ead_schedule)))
        # Monthly discount factor
        r_m = inp.effective_rate[i] / 12.0
        t_idx = np.arange(1, len(marginals) + 1)
        dfm = 1.0 / (1.0 + r_m) ** t_idx
        ecl_lifetime[i] = float(np.sum(marginals * lgd_eff[i] * ead_schedule * dfm))
        pd_lifetime_vec[i] = lifetime_pd(inp.pd_12m[i], rm)

    reported = np.where(stages == 1, ecl_12m, ecl_lifetime)

    return pd.DataFrame(
        {
            "loan_id": inp.loan_id,
            "as_of_date": pd.to_datetime(inp.as_of_date).date
                if not isinstance(inp.as_of_date[0], (pd.Timestamp,)) else inp.as_of_date,
            "scenario": scenario,
            "pd_12m": inp.pd_12m.astype(float).round(8),
            "pd_lifetime": pd_lifetime_vec.round(8),
            "lgd": lgd_eff.round(8),
            "ead": np.asarray(inp.ead, dtype=float).round(2),
            "ecl_12m": ecl_12m.round(4),
            "ecl_lifetime": ecl_lifetime.round(4),
            "reported_ecl": np.round(reported, 4),
            "stage": stages.astype(int),
            "model_version": model_version,
        }
    )


# ---------------------------------------------------------------------
# Portfolio-level helpers
# ---------------------------------------------------------------------
def portfolio_summary(ecl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate: reported ECL, count, mean PD, mean LGD, by stage."""
    return (
        ecl_df.groupby("stage")
        .agg(
            loans=("loan_id", "size"),
            reported_ecl=("reported_ecl", "sum"),
            mean_pd_12m=("pd_12m", "mean"),
            mean_lgd=("lgd", "mean"),
            mean_ead=("ead", "mean"),
        )
        .reset_index()
    )


def portfolio_by(df: pd.DataFrame, by: str, ecl_col: str = "reported_ecl") -> pd.DataFrame:
    return (df.groupby(by).agg(loans=("loan_id", "size"),
                                reported_ecl=(ecl_col, "sum")).reset_index())
