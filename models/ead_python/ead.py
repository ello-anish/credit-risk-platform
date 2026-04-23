"""EAD Python — outstanding principal at default.

LendingClub loans are term installment loans, so **EAD = outstanding principal
balance at the moment of default**. There is no undrawn commitment, so the
Credit Conversion Factor (CCF) regression that would otherwise apply to
revolving credit (credit cards / HELOCs / LOCs) is NOT applicable here.

If the portfolio later expanded to include revolving exposures, the standard
approach would be:

    EAD = drawn_balance + CCF * undrawn_commitment

where CCF is regressed on macro + exposure characteristics. That is left as
a documented extension rather than a live feature.

Linear-amortisation rule used for ``predict_ead``:
    principal_remaining(t) = funded_amnt * max(0, 1 - t / term_months)

This is the closed-form solution for a fully-amortising fixed-term loan
assuming equal scheduled principal. LendingClub loans in practice follow an
annuity schedule (constant total payment, interest front-loaded) so this
underestimates principal in early months by ~2-5 %. For small-footprint
Tier 1 ECL this approximation is acceptable; a full annuity amortisation
helper is provided below as ``annuity_principal_remaining``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_ead_linear(
    funded_amnt: pd.Series,
    term_months: pd.Series,
    months_elapsed: pd.Series,
) -> pd.Series:
    """Linear (straight-line) amortisation — simple and conservative."""
    frac = 1.0 - (months_elapsed.astype(float) / term_months.astype(float))
    frac = frac.clip(lower=0.0, upper=1.0)
    return (funded_amnt.astype(float) * frac).round(2)


def annuity_principal_remaining(
    funded_amnt: pd.Series,
    term_months: pd.Series,
    int_rate_annual: pd.Series,
    months_elapsed: pd.Series,
) -> pd.Series:
    """Outstanding principal under a constant-payment annuity amortisation.

    Formula (standard):
        balance(t) = P * (1+r)^t - A * ((1+r)^t - 1)/r
    where r = monthly rate, A = scheduled monthly payment, P = original principal.
    """
    r = int_rate_annual.astype(float) / 12.0
    n = term_months.astype(float)
    t = months_elapsed.astype(float).clip(lower=0.0, upper=n)
    P = funded_amnt.astype(float)
    # Monthly payment A = P * r / (1 - (1+r)^-n)
    # Guard r -> 0 with a linear fallback
    linear = predict_ead_linear(funded_amnt, term_months, months_elapsed)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(r > 0,
                     P * r / (1.0 - np.power(1.0 + r, -n)),
                     P / n)
        factor_t = np.power(1.0 + r, t)
        bal = np.where(r > 0,
                       P * factor_t - A * (factor_t - 1.0) / r,
                       P * (1.0 - t / n))
    out = pd.Series(bal, index=P.index).round(2).clip(lower=0.0)
    # Where either rate or term is zero/NaN, fall back to linear
    out = out.where(out.notna() & (out >= 0), linear)
    return out


def predict_ead(loans: pd.DataFrame, months_elapsed_col: str = "months_elapsed",
                method: str = "annuity") -> pd.Series:
    """Dispatch helper — ``method`` in ``{"annuity", "linear"}``."""
    if method == "annuity":
        return annuity_principal_remaining(
            loans["funded_amnt"], loans["term_months"],
            loans["int_rate"], loans[months_elapsed_col],
        )
    return predict_ead_linear(
        loans["funded_amnt"], loans["term_months"], loans[months_elapsed_col]
    )
