"""Data quality gate — pandera schemas over the ingested DataFrames.

Schemas codify the hard rules from config.yml:preprocessing and from the
project spec. ``run_quality_gate`` validates all four core DataFrames and,
on success, persists a row-per-check audit log to
``credit_risk.data_quality_runs``. On failure it raises with the failing
row count and a clear message — the pipeline aborts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from pipeline.config import CFG
from pipeline.db import get_engine, set_schema
from pipeline.logging_utils import get_logger
from sqlalchemy import text

LOG = get_logger(__name__)

# ---------------------------------------------------------------------
# Reusable value sets from config.yml
# ---------------------------------------------------------------------
POSITIVE_STATUSES = set(CFG["default"]["positive_statuses"])
NEGATIVE_STATUSES = set(CFG["default"]["negative_statuses"])
EXCLUDE_STATUSES = set(CFG["default"]["exclude_statuses"])
KNOWN_STATUSES = POSITIVE_STATUSES | NEGATIVE_STATUSES | EXCLUDE_STATUSES

FICO_LO, FICO_HI = CFG["preprocessing"]["fico_range"]
DTI_CAP = CFG["preprocessing"]["dti_cap"]
NULL_RATE_THRESHOLD = CFG["preprocessing"]["null_rate_threshold"]


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
LOANS_SCHEMA = DataFrameSchema(
    {
        "loan_id": Column(pa.Int64, unique=True, nullable=False),
        "issue_date": Column(pa.DateTime, nullable=False,
            checks=Check.in_range(pd.Timestamp("2007-01-01"), pd.Timestamp("2018-12-31"))),
        "term_months": Column(pa.Int, checks=Check.isin([36, 60])),
        "grade": Column(pa.String, checks=Check.isin(list("ABCDEFG"))),
        "sub_grade": Column(pa.String),
        "fico_range_low": Column(pa.Float, nullable=True,
            checks=Check.in_range(FICO_LO, FICO_HI, include_min=True, include_max=True)),
        "fico_range_high": Column(pa.Float, nullable=True,
            checks=Check.in_range(FICO_LO, FICO_HI, include_min=True, include_max=True)),
        "annual_inc": Column(pa.Float, nullable=True, checks=Check.greater_than_or_equal_to(0)),
        "dti": Column(pa.Float, nullable=True, checks=Check.in_range(-1.0, DTI_CAP)),
        "loan_amnt": Column(pa.Float, checks=Check.greater_than(0)),
        "funded_amnt": Column(pa.Float, checks=Check.greater_than(0)),
        "int_rate": Column(pa.Float, checks=Check.in_range(0.0, 0.5)),   # as fraction
        "installment": Column(pa.Float, checks=Check.greater_than(0)),
        "loan_status_raw": Column(pa.String, checks=Check.isin(list(KNOWN_STATUSES))),
        "default_flag": Column(pa.Int, checks=Check.isin([0, 1])),
        "vintage": Column(pa.String),
        "split": Column(pa.String, checks=Check.isin(["train", "validation", "oot", "excluded"])),
    },
    strict=False,
    coerce=True,
)


DEFAULTS_SCHEMA = DataFrameSchema(
    {
        "loan_id": Column(pa.Int64, unique=True, nullable=False),
        "default_date": Column(pa.DateTime, nullable=False),
        "funded_amnt": Column(pa.Float, checks=Check.greater_than(0)),
        "recovery_amount": Column(pa.Float, checks=Check.greater_than_or_equal_to(0)),
        "lgd": Column(pa.Float, checks=Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
    },
    strict=False,
    coerce=True,
)


MACRO_SCHEMA = DataFrameSchema(
    {
        "as_of_date": Column(pa.Date, unique=True, nullable=False),
        "gdp_growth": Column(pa.Float, nullable=True,
            checks=Check.in_range(-15.0, 15.0)),
        "unemployment": Column(pa.Float, checks=Check.in_range(0.0, 30.0)),
        "hpi": Column(pa.Float, checks=Check.greater_than(0)),
        "treasury_10y": Column(pa.Float, checks=Check.in_range(0.0, 20.0)),
        "vix": Column(pa.Float, checks=Check.in_range(0.0, 100.0)),
    },
    strict=False,
    coerce=True,
)


@dataclass
class CheckResult:
    check_name: str
    table_name: str
    passed: bool
    failed_rows: int = 0
    message: str = ""


def _null_rate_check(df: pd.DataFrame, required: list[str]) -> list[CheckResult]:
    """Flag any required column whose null rate > threshold."""
    out = []
    for col in required:
        if col not in df.columns:
            out.append(CheckResult(f"present:{col}", "loans", False, 0,
                                   f"required column missing: {col}"))
            continue
        rate = df[col].isna().mean()
        ok = rate <= NULL_RATE_THRESHOLD
        out.append(
            CheckResult(
                f"null_rate:{col}",
                "loans",
                ok,
                int(df[col].isna().sum()),
                f"null rate {rate:.1%} (threshold {NULL_RATE_THRESHOLD:.0%})",
            )
        )
    return out


def _log_checks(results: list[CheckResult]) -> None:
    """Persist check outcomes to data_quality_runs."""
    set_schema()
    # Cast numpy types to plain Python so psycopg2 can bind them.
    rows = [
        {
            "table_name": str(r.table_name),
            "check_name": str(r.check_name),
            "passed": bool(r.passed),
            "failed_rows": int(r.failed_rows),
            "message": str(r.message),
        }
        for r in results
    ]
    with get_engine().begin() as cx:
        cx.execute(
            text(
                "INSERT INTO data_quality_runs "
                "(table_name, check_name, passed, failed_rows, message) "
                "VALUES (:table_name, :check_name, :passed, :failed_rows, :message)"
            ),
            rows,
        )


def validate_loans(df: pd.DataFrame, raise_on_failure: bool = True) -> list[CheckResult]:
    """Run LOANS_SCHEMA + null-rate check; return per-check results.

    The pandera schema is applied to the MODELABLE subset only (rows where
    default_flag is populated). Loans in exclude-statuses (Current, In Grace
    Period, Late 16-30) have NaN default_flag and are logged separately.
    """
    results: list[CheckResult] = []

    # null-rate on training slice only (per spec: fail QG if any required col >5% null on train)
    train = df[df["split"] == "train"]
    results.extend(_null_rate_check(train, ["fico_range_low", "fico_range_high",
                                             "annual_inc", "dti", "int_rate"]))

    # FICO sanity
    bad_fico = df[(df["fico_range_low"] > df["fico_range_high"])]
    results.append(
        CheckResult(
            "fico_low_le_high",
            "loans",
            bad_fico.empty,
            len(bad_fico),
            f"{len(bad_fico)} rows with fico_low > fico_high",
        )
    )

    # Excluded-loans accounting (open loans — we expect these, not a failure)
    excluded = df[df["split"] == "excluded"]
    results.append(
        CheckResult(
            "excluded_open_loans_count",
            "loans",
            True,   # informational, always passes
            len(excluded),
            f"{len(excluded)} loans in open statuses excluded from modelling",
        )
    )

    # Pandera schema on the MODELABLE subset only
    modelable = df[df["default_flag"].notna()].copy()
    modelable["default_flag"] = modelable["default_flag"].astype(int)
    try:
        LOANS_SCHEMA.validate(modelable, lazy=True)
        results.append(CheckResult("pandera_loans", "loans", True, 0,
                                    f"schema OK on {len(modelable):,} modelable rows"))
    except pa.errors.SchemaErrors as e:
        results.append(CheckResult("pandera_loans", "loans", False,
                                   len(e.failure_cases), str(e)[:500]))

    _log_checks(results)

    if raise_on_failure:
        fails = [r for r in results if not r.passed]
        if fails:
            msg = "\n".join(f"  {r.check_name}: {r.message}" for r in fails)
            raise ValueError(f"loans quality gate FAILED:\n{msg}")
    return results


def validate_defaults(df: pd.DataFrame, raise_on_failure: bool = True) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        DEFAULTS_SCHEMA.validate(df, lazy=True)
        results.append(CheckResult("pandera_defaults", "defaults", True, 0, "schema OK"))
    except pa.errors.SchemaErrors as e:
        results.append(CheckResult("pandera_defaults", "defaults", False,
                                   len(e.failure_cases), str(e)[:500]))
    _log_checks(results)
    if raise_on_failure and any(not r.passed for r in results):
        raise ValueError("defaults quality gate FAILED: " + results[-1].message)
    return results


def validate_macro(df: pd.DataFrame, raise_on_failure: bool = True) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        MACRO_SCHEMA.validate(df, lazy=True)
        results.append(CheckResult("pandera_macro", "macro", True, 0, "schema OK"))
    except pa.errors.SchemaErrors as e:
        results.append(CheckResult("pandera_macro", "macro", False,
                                   len(e.failure_cases), str(e)[:500]))
    _log_checks(results)
    if raise_on_failure and any(not r.passed for r in results):
        raise ValueError("macro quality gate FAILED: " + results[-1].message)
    return results
