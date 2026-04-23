"""Vasicek one-factor link between macro and portfolio PD.

We fit a simple logistic of realised default rate per vintage-quarter on the
macro factors. The resulting beta vector is used to shift log-odds of each
loan's PD under a scenario shock.

The math:
    logit(DR_{vq}) = alpha + beta_u * d_unemp_{vq} + beta_g * gdp_g_{vq} + beta_h * hpi_c_{vq}

Given a shocked macro vector m*, the implied log-odds delta vs baseline is:
    delta = beta · (m* - m_baseline)
and each loan's PD is shifted by:
    pd_shocked = sigmoid( logit(pd) + delta )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pipeline.config import CFG
from pipeline.logging_utils import get_logger

LOG = get_logger(__name__)


@dataclass
class VasicekFit:
    alpha: float
    beta_unemp: float
    beta_gdp: float
    beta_hpi: float
    r_squared: float
    n: int

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta_unemp": self.beta_unemp,
            "beta_gdp": self.beta_gdp,
            "beta_hpi": self.beta_hpi,
            "r_squared": self.r_squared,
            "n": self.n,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_vasicek(loans: pd.DataFrame, macro: pd.DataFrame) -> VasicekFit:
    """Fit the Vasicek link on vintage-quarter default rates vs macro.

    Args:
        loans: must contain issue_date, default_flag, split.
        macro: must contain as_of_date, unemployment, gdp_growth, hpi.
    """
    lc = loans.dropna(subset=["default_flag"]).copy()
    lc["as_of_date"] = (pd.to_datetime(lc["issue_date"])
                        .dt.to_period("Q").dt.end_time.dt.normalize())
    agg = (lc.groupby("as_of_date")
           .agg(defaults=("default_flag", "sum"), n=("default_flag", "size"))
           .reset_index())
    agg["dr"] = agg["defaults"] / agg["n"]
    agg = agg[agg["n"] >= 50]   # exclude very small quarters

    m = macro.copy()
    m["as_of_date"] = pd.to_datetime(m["as_of_date"])
    m = m.sort_values("as_of_date").reset_index(drop=True)
    # YoY deltas
    m["unemployment_delta"] = m["unemployment"].diff(4)
    m["hpi_change"] = m["hpi"].pct_change(4) * 100.0

    j = (agg.merge(
        m[["as_of_date", "unemployment_delta", "gdp_growth", "hpi_change"]],
        on="as_of_date", how="inner").dropna())
    if len(j) < 6:
        LOG.warning("Only %d vintage-quarters after join — Vasicek fit may be unstable",
                    len(j))

    y = _logit(j["dr"].values)
    X = j[["unemployment_delta", "gdp_growth", "hpi_change"]].values
    X = sm.add_constant(X)
    res = sm.OLS(y, X).fit()

    return VasicekFit(
        alpha=float(res.params[0]),
        beta_unemp=float(res.params[1]),
        beta_gdp=float(res.params[2]),
        beta_hpi=float(res.params[3]),
        r_squared=float(res.rsquared),
        n=int(len(j)),
    )


def shift_pd(pd_baseline: np.ndarray, fit: VasicekFit, shock: dict) -> np.ndarray:
    """Apply a macro shock to baseline PDs.

    ``shock`` keys: ``unemployment_delta``, ``gdp_growth``, ``hpi_change``. Any
    missing key is treated as 0. The shift is in logit space.
    """
    delta = (
        fit.beta_unemp * float(shock.get("unemployment_delta", 0.0))
        + fit.beta_gdp * float(shock.get("gdp_growth", 0.0))
        + fit.beta_hpi * float(shock.get("hpi_change", 0.0))
    )
    return _sigmoid(_logit(pd_baseline) + delta)
