"""Fetch FRED macroeconomic series and cache to ``data/raw/macro.parquet``.

Series pulled:
    GDPC1    — Real GDP (quarterly), converted to YoY % growth
    UNRATE   — Civilian unemployment rate (monthly, %)
    CSUSHPISA — Case-Shiller HPI (monthly, level)
    DGS10    — 10Y Treasury yield (daily, %)
    VIXCLS   — VIX close (daily)

Output columns (quarterly, end-of-quarter):
    as_of_date, gdp_growth, unemployment, hpi, treasury_10y, vix

NO synthetic fallback. If FRED is unreachable and no cached file exists,
the function halts with an actionable error message.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import CFG, REPO_ROOT
from pipeline.logging_utils import get_logger

LOG = get_logger(__name__)

FRED_SERIES = {
    "GDPC1": "gdp_real",       # chained, real
    "UNRATE": "unemployment",
    "CSUSHPISA": "hpi",
    "DGS10": "treasury_10y",
    "VIXCLS": "vix",
}

START_DATE = "2006-01-01"   # covers LendingClub horizon + 1yr pre-window
END_DATE = "2020-12-31"


def _cache_path() -> Path:
    return REPO_ROOT / CFG["paths"]["macro_data"]


def _try_fetch_fred() -> pd.DataFrame | None:
    """Pull FRED via pandas_datareader. Returns None on any network error."""
    try:
        from pandas_datareader import data as pdr  # local import
    except ImportError:
        LOG.error("pandas_datareader not installed. pip install -r requirements.txt")
        return None

    frames = {}
    for code in FRED_SERIES:
        try:
            s = pdr.DataReader(code, "fred", START_DATE, END_DATE)
            frames[code] = s
            LOG.info("FRED %s: %d observations", code, len(s))
        except Exception as e:  # noqa: BLE001
            LOG.error("FRED fetch failed for %s: %s", code, e)
            return None
    return _align_to_quarters(frames)


def _align_to_quarters(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Resample all series to quarter-end and compute YoY real-GDP growth."""
    # GDPC1 is quarterly; CSUSHPISA monthly; UNRATE monthly; DGS10 + VIX daily.
    gdp = frames["GDPC1"].resample("QE").last()
    gdp["gdp_growth"] = gdp["GDPC1"].pct_change(4) * 100.0  # YoY %

    unemp = frames["UNRATE"].resample("QE").mean()
    hpi = frames["CSUSHPISA"].resample("QE").last()
    y10 = frames["DGS10"].resample("QE").mean()
    vix = frames["VIXCLS"].resample("QE").mean()

    out = pd.concat(
        [
            gdp["gdp_growth"],
            unemp["UNRATE"].rename("unemployment"),
            hpi["CSUSHPISA"].rename("hpi"),
            y10["DGS10"].rename("treasury_10y"),
            vix["VIXCLS"].rename("vix"),
        ],
        axis=1,
    )
    out.index.name = "as_of_date"
    out = out.reset_index()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"]).dt.date
    return out.dropna(subset=["unemployment", "hpi"]).reset_index(drop=True)


def get_macro(force_refresh: bool = False) -> pd.DataFrame:
    """Return the quarterly macro DataFrame, caching to ``data/raw/macro.parquet``.

    Precedence:
        1. If cache exists and force_refresh=False → load cache.
        2. Else fetch from FRED and write cache.
        3. Else HALT with actionable message (no synthetic fallback).
    """
    cache = _cache_path()
    if cache.exists() and not force_refresh:
        LOG.info("Loading cached macro data: %s", cache)
        return pd.read_parquet(cache)

    LOG.info("Fetching macro data from FRED...")
    df = _try_fetch_fred()
    if df is None or df.empty:
        raise RuntimeError(
            "Macro fetch failed. No synthetic fallback per project policy.\n"
            "Manual workaround: download CSVs from https://fred.stlouisfed.org/ for "
            "GDPC1, UNRATE, CSUSHPISA, DGS10, VIXCLS, build a quarterly-aligned parquet, "
            f"and save it as: {cache}\n"
            "Expected columns: as_of_date, gdp_growth, unemployment, hpi, treasury_10y, vix"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, compression="snappy")
    LOG.info("Wrote %d quarterly rows to %s", len(df), cache)
    return df


if __name__ == "__main__":
    df = get_macro(force_refresh=False)
    print(df.tail(10).to_string(index=False))
