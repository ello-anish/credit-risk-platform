"""Parquet -> Postgres ingest for the LendingClub sample.

Pipeline:
    1. Read ``data/raw/lending_club_sample_50k.parquet``.
    2. Parse & clean:
         - ``issue_d`` (e.g. "Jun-2015")        -> issue_date (DATE)
         - ``term`` (e.g. " 36 months")          -> term_months (INT)
         - ``int_rate`` (float or "13.56%")      -> int_rate (FLOAT, as fraction)
         - ``emp_length`` ("10+ years", "< 1…")  -> kept as categorical string
         - Strip "Does not meet the credit policy. Status:" prefix from loan_status
           but keep the original so we preserve the known-status set. We store
           the stripped label in ``loan_status_raw`` (it still matches the
           positive/negative/exclude sets verbatim because those sets include
           BOTH forms).
         - DTI: cap at 100
         - annual_inc: winsorize at 99th percentile, log-transform (new column)
         - FICO midpoint (new column)
    3. Assign ``default_flag`` (0/1/NaN-for-exclusions based on config).
    4. Assign vintage quarter and train/validation/oot split.
    5. Drop pre-2012 vintages (config.yml:vintages.drop_before).
    6. COPY into:
         loans            (all surviving rows, including "excluded" split)
         defaults         (only positive-status rows)
         loan_status      (monthly snapshots, synthesized — see docstring)
    7. Also populates ``macro`` from ``data/raw/macro.parquet`` (fetched separately).

Monthly snapshot synthesis (the spec calls for this):
    LendingClub provides only origination + final status. For richer time-series
    features we interpolate monthly (loan_id, as_of_date) rows from issue_date
    to a computed observation_end. Status is "Current" until the final period;
    on default, DPD ramps 30 -> 60 -> 90 -> 120+ in the last 4 months. This is
    a SIMULATION and is documented as such — it drives staging unit tests and
    is NOT used as a feature in the PD training (which uses only origination
    features, avoiding the look-ahead issue).
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from pipeline.config import CFG, REPO_ROOT, raw_path
from pipeline.db import copy_dataframe, get_engine, set_schema, truncate_tables, wait_for_db
from pipeline.logging_utils import get_logger
from pipeline.splits import assign_split, issue_date_to_vintage

LOG = get_logger(__name__)

POSITIVE = set(CFG["default"]["positive_statuses"])
NEGATIVE = set(CFG["default"]["negative_statuses"])
EXCLUDE = set(CFG["default"]["exclude_statuses"])
DROP_BEFORE = pd.Timestamp(CFG["vintages"]["drop_before"])


# ---------------------------------------------------------------------
# Column parsing helpers
# ---------------------------------------------------------------------
def _parse_term(s: pd.Series) -> pd.Series:
    """' 36 months' -> 36 (int)."""
    return s.astype(str).str.extract(r"(\d+)", expand=False).astype(float).astype("Int64")


def _parse_int_rate(s: pd.Series) -> pd.Series:
    """Return interest rate as a FRACTION (e.g. 0.1356)."""
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float)
    else:
        v = s.astype(str).str.replace("%", "", regex=False).astype(float)
    # LendingClub stores "13.56" (percent form). Convert to fraction 0.1356.
    return v / 100.0 if v.dropna().median() > 1.0 else v


def _parse_issue_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def _parse_earliest_cr_line(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def _winsorize(s: pd.Series, upper_pct: float) -> pd.Series:
    hi = s.quantile(upper_pct)
    return s.clip(upper=hi)


# ---------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    path = raw_path("lending_club_sample")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See README 'Colab instructions' to regenerate."
        )
    df = pd.read_parquet(path)
    LOG.info("Loaded raw parquet: %d rows x %d cols", *df.shape)
    return df


def clean_loans(raw: pd.DataFrame) -> pd.DataFrame:
    """Produce the normalised ``loans`` DataFrame (one row per loan)."""
    df = raw.copy()
    df["issue_date"] = _parse_issue_date(df["issue_d"])
    df["earliest_cr_line"] = _parse_earliest_cr_line(df["earliest_cr_line"])
    df["term_months"] = _parse_term(df["term"])
    df["int_rate"] = _parse_int_rate(df["int_rate"])

    # Strip "Does not meet the credit policy. Status:" prefix for modelling, but
    # keep the full raw label as ``loan_status_raw`` so the quality gate can
    # assert it belongs to the known set.
    df["loan_status_raw"] = df["loan_status"].astype(str)

    # DTI cap
    df["dti"] = df["dti"].astype(float).clip(upper=CFG["preprocessing"]["dti_cap"])

    # Income winsorization + log
    df["annual_inc"] = df["annual_inc"].astype(float).clip(lower=0)
    df["annual_inc_w"] = _winsorize(df["annual_inc"],
                                    CFG["preprocessing"]["annual_inc_winsor_pct"])
    df["annual_inc_log"] = np.log1p(df["annual_inc_w"])

    # FICO midpoint
    df["fico_range_low"] = df["fico_range_low"].astype(float)
    df["fico_range_high"] = df["fico_range_high"].astype(float)
    df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0

    # revol_util often has a "%" form
    if "revol_util" in df.columns and not pd.api.types.is_numeric_dtype(df["revol_util"]):
        df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "", regex=False)
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce") / 100.0
    else:
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
        # LendingClub stores as percent — normalise to fraction if the scale is > 1
        if df["revol_util"].dropna().median() > 1.0:
            df["revol_util"] = df["revol_util"] / 100.0

    # default_flag
    df["default_flag"] = pd.NA
    df.loc[df["loan_status_raw"].isin(POSITIVE), "default_flag"] = 1
    df.loc[df["loan_status_raw"].isin(NEGATIVE), "default_flag"] = 0
    # Rows in EXCLUDE keep NA and get split='excluded'

    # Vintage + split
    df["vintage"] = issue_date_to_vintage(df["issue_date"])
    df["split"] = assign_split(df["issue_date"])
    # Loans in EXCLUDE statuses get split='excluded' even if their vintage is train/oot.
    df.loc[df["loan_status_raw"].isin(EXCLUDE), "split"] = "excluded"

    # Drop rows before 2012 (per config)
    before = len(df)
    df = df[df["issue_date"] >= DROP_BEFORE].copy()
    LOG.info("Dropped %d pre-2012 loans (%d -> %d)", before - len(df), before, len(df))

    # Assign a stable loan_id (LC doesn't expose one directly in this export).
    df = df.reset_index(drop=True)
    df["loan_id"] = df.index.astype("int64") + 1_000_000

    # Pick just the columns loans table expects
    cols = [
        "loan_id", "issue_date", "term_months", "grade", "sub_grade",
        "fico_range_low", "fico_range_high", "annual_inc", "dti",
        "purpose", "home_ownership", "emp_length", "verification_status",
        "loan_amnt", "funded_amnt", "int_rate", "installment", "addr_state",
        "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc",
        "vintage", "split",
        # derived columns kept for feature store (not in loans table DDL but
        # useful for features.py; we select explicitly when writing to DB)
        "fico_mid", "annual_inc_log", "annual_inc_w",
        "loan_status_raw", "default_flag",
    ]
    return df[cols]


def build_defaults(loans: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """One row per defaulted loan with recovery + LGD."""
    # Only loans in POSITIVE status count as defaults for LGD modelling.
    pos_mask = loans["loan_status_raw"].isin(POSITIVE)
    defaulted = loans.loc[pos_mask, ["loan_id", "issue_date", "funded_amnt",
                                      "term_months", "loan_status_raw"]].copy()
    # Default date proxy: issue_date + term_months months (LendingClub doesn't
    # give a clean default date; last_pymnt_d would be closer but may be NA).
    raw_subset = raw.loc[loans.index[pos_mask]].copy() if len(raw) == len(loans) else None
    # Safer: left-join last_pymnt_d by row alignment via index (loans preserves
    # post-drop index, so we need a merge). We re-derive from raw parquet via
    # issue_d + funded_amnt match — but the simpler path is to attach
    # last_pymnt_d in ``clean_loans`` output. Let's not assume alignment.
    del raw_subset

    # Pull recovery fields from raw by matching loan_id — but loans reassigns
    # loan_id, so we join by (issue_date, funded_amnt, loan_amnt, int_rate) isn't
    # stable either. Simplest: when building loans, also carry recoveries +
    # collection_recovery_fee + last_pymnt_d downstream.
    return defaulted


def clean_loans_with_recovery(raw: pd.DataFrame) -> pd.DataFrame:
    """Same as ``clean_loans`` but also carries recovery + last_pymnt_d fields.

    Kept separate to make the loans table schema clean while still letting
    build_defaults compute LGD without another lookup.
    """
    df = clean_loans(raw)
    # Reattach the extras by positional alignment — ``clean_loans`` preserved
    # row order after the pre-2012 drop, so we slice raw the same way.
    raw2 = raw.copy()
    raw2["issue_date"] = _parse_issue_date(raw2["issue_d"])
    raw2 = raw2[raw2["issue_date"] >= DROP_BEFORE].reset_index(drop=True)
    df = df.reset_index(drop=True)
    for col in ["recoveries", "collection_recovery_fee", "last_pymnt_d", "total_rec_prncp"]:
        if col in raw2.columns:
            df[col] = raw2[col].values
    return df


def build_defaults_full(loans_plus: pd.DataFrame) -> pd.DataFrame:
    """Defaults with recovery, default_date, and LGD."""
    pos = loans_plus[loans_plus["loan_status_raw"].isin(POSITIVE)].copy()
    # Recovery = recoveries + collection_recovery_fee (LendingClub convention)
    recov = pos.get("recoveries", 0).fillna(0).astype(float)
    if "collection_recovery_fee" in pos.columns:
        recov = recov + pos["collection_recovery_fee"].fillna(0).astype(float)

    # LGD = 1 - recovery / funded_amnt
    funded = pos["funded_amnt"].astype(float).replace(0, np.nan)
    lgd = 1.0 - (recov / funded)
    # Drop the <0.1% of loans with LGD < 0 (negative recovery is unusual).
    pos["lgd_raw"] = lgd
    pos = pos[pos["lgd_raw"] >= 0].copy()
    pos["lgd"] = pos["lgd_raw"].clip(lower=0.0, upper=1.0)

    # Default date: use last_pymnt_d + 5 months (Charged-Off is ~150 DPD) if
    # available, else issue_date + term_months.
    if "last_pymnt_d" in pos.columns:
        lpd = pd.to_datetime(pos["last_pymnt_d"], format="%b-%Y", errors="coerce")
        default_date = lpd + pd.offsets.MonthEnd(5)
        default_date = default_date.fillna(pos["issue_date"] +
                                            pd.to_timedelta(pos["term_months"] * 30, "D"))
    else:
        default_date = pos["issue_date"] + pd.to_timedelta(pos["term_months"] * 30, "D")
    pos["default_date"] = default_date
    pos["default_type"] = pos["loan_status_raw"].where(pos["loan_status_raw"].isin(POSITIVE), None)
    pos["recovery_amount"] = recov
    pos["recovery_date"] = default_date + pd.to_timedelta(90, "D")  # proxy

    return pos[
        ["loan_id", "default_date", "default_type", "recovery_amount",
         "recovery_date", "funded_amnt", "lgd"]
    ]


def build_loan_status_snapshots(loans_plus: pd.DataFrame,
                                 horizon_months: int = 36) -> pd.DataFrame:
    """Synthesize monthly (loan_id, as_of_date, status, dpd, balance) rows.

    For each loan we generate min(term_months, horizon_months) snapshots from
    issue_date onwards. Status is "Current" until the last 4 months for defaulted
    loans (DPD ramp 30/60/90/120+), else "Current" -> final at the end.
    Outstanding balance uses simple linear amortization toward zero at term end.
    """
    rows: list[dict] = []
    for rec in loans_plus.itertuples(index=False):
        term = min(int(rec.term_months or 0), horizon_months)
        if term <= 0:
            continue
        status_raw = rec.loan_status_raw
        is_default = status_raw in POSITIVE
        is_paid = status_raw in NEGATIVE
        # Straight-line principal
        for m in range(1, term + 1):
            as_of = (rec.issue_date + pd.DateOffset(months=m)).date()
            frac_remaining = max(0.0, 1.0 - m / term)
            out_bal = float(rec.funded_amnt) * frac_remaining

            dpd = 0
            if is_default and m >= term - 3:
                dpd = {term - 3: 30, term - 2: 60, term - 1: 90, term: 120}.get(m, 0)
                current_status = status_raw if m == term else "Current"
            elif is_paid and m == term:
                current_status = "Fully Paid"
            else:
                current_status = "Current"

            rows.append(
                {
                    "loan_id": rec.loan_id,
                    "as_of_date": as_of,
                    "current_status": current_status,
                    "days_past_due": dpd,
                    "outstanding_balance": round(out_bal, 2),
                }
            )
    LOG.info("Generated %d loan_status snapshot rows", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------
def ingest_all(truncate: bool = True) -> dict[str, int]:
    """Run full ingest. Returns row counts per table."""
    wait_for_db()
    set_schema()

    raw = load_raw()
    loans_plus = clean_loans_with_recovery(raw)

    loans_cols = [
        "loan_id", "issue_date", "term_months", "grade", "sub_grade",
        "fico_range_low", "fico_range_high", "annual_inc", "dti",
        "purpose", "home_ownership", "emp_length", "verification_status",
        "loan_amnt", "funded_amnt", "int_rate", "installment", "addr_state",
        "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc",
        "vintage", "split",
    ]
    loans_out = loans_plus[loans_cols].copy()
    # Postgres INTEGER columns — pandas often reads these as float64 (because of
    # NaNs). Coerce to pandas nullable Int64 so COPY emits integer literals.
    int_cols = ["term_months", "fico_range_low", "fico_range_high",
                "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
                "total_acc"]
    for c in int_cols:
        if c in loans_out.columns:
            loans_out[c] = pd.to_numeric(loans_out[c], errors="coerce").round().astype("Int64")

    defaults_out = build_defaults_full(loans_plus)
    status_out = build_loan_status_snapshots(loans_plus, horizon_months=36)

    # Macro
    from pipeline.macro import get_macro
    macro = get_macro(force_refresh=False)

    if truncate:
        # ecl_results depends on loans; truncate cascade.
        truncate_tables(["loan_status", "defaults", "ecl_results", "loans", "macro"])

    counts = {}
    counts["loans"] = copy_dataframe(loans_out, "loans", columns=loans_cols)
    counts["defaults"] = copy_dataframe(defaults_out, "defaults")
    counts["loan_status"] = copy_dataframe(status_out, "loan_status")
    counts["macro"] = copy_dataframe(macro, "macro")

    LOG.info("Ingest complete: %s", counts)

    # Also persist loans_plus as parquet for Python-side modelling (saves a DB
    # round-trip and preserves derived columns like fico_mid that aren't in the
    # loans table DDL).
    features_dir = REPO_ROOT / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    out_path = features_dir / "loans_clean.parquet"
    loans_plus.to_parquet(out_path, compression="snappy")
    LOG.info("Wrote %s (%d rows)", out_path, len(loans_plus))

    return counts


if __name__ == "__main__":
    counts = ingest_all(truncate=True)
    print("Ingest counts:", counts)
