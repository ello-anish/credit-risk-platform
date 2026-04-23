"""Data quality: synthetic bad DataFrames should fail the pandera gates."""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest

from pipeline.quality import LOANS_SCHEMA, DEFAULTS_SCHEMA, MACRO_SCHEMA


def _minimal_good_loan_row() -> dict:
    return {
        "loan_id": 1,
        "issue_date": pd.Timestamp("2015-06-01"),
        "term_months": 36,
        "grade": "B",
        "sub_grade": "B3",
        "fico_range_low": 700.0,
        "fico_range_high": 704.0,
        "annual_inc": 60000.0,
        "dti": 15.0,
        "loan_amnt": 10000.0,
        "funded_amnt": 10000.0,
        "int_rate": 0.12,
        "installment": 332.0,
        "loan_status_raw": "Fully Paid",
        "default_flag": 0,
        "vintage": "2015-Q2",
        "split": "train",
    }


def test_good_loan_row_passes():
    df = pd.DataFrame([_minimal_good_loan_row()])
    LOANS_SCHEMA.validate(df)   # must not raise


def test_fico_out_of_range_fails():
    bad = _minimal_good_loan_row()
    bad["fico_range_low"] = 100.0          # below valid range
    with pytest.raises(pa.errors.SchemaError):
        LOANS_SCHEMA.validate(pd.DataFrame([bad]))


def test_grade_not_in_set_fails():
    bad = _minimal_good_loan_row()
    bad["grade"] = "Z"
    with pytest.raises(pa.errors.SchemaError):
        LOANS_SCHEMA.validate(pd.DataFrame([bad]))


def test_negative_int_rate_fails():
    bad = _minimal_good_loan_row()
    bad["int_rate"] = -0.01
    with pytest.raises(pa.errors.SchemaError):
        LOANS_SCHEMA.validate(pd.DataFrame([bad]))


def test_unknown_loan_status_fails():
    bad = _minimal_good_loan_row()
    bad["loan_status_raw"] = "Foo Bar"
    with pytest.raises(pa.errors.SchemaError):
        LOANS_SCHEMA.validate(pd.DataFrame([bad]))


def test_defaults_schema_rejects_lgd_above_one():
    bad = pd.DataFrame([{
        "loan_id": 1,
        "default_date": pd.Timestamp("2016-01-01"),
        "funded_amnt": 10000.0,
        "recovery_amount": 0.0,
        "lgd": 1.5,
    }])
    with pytest.raises(pa.errors.SchemaError):
        DEFAULTS_SCHEMA.validate(bad)


def test_macro_schema_rejects_impossible_unemployment():
    bad = pd.DataFrame([{
        "as_of_date": pd.Timestamp("2015-03-31").date(),
        "gdp_growth": 2.0,
        "unemployment": 55.0,       # impossibly high
        "hpi": 200.0,
        "treasury_10y": 2.5,
        "vix": 15.0,
    }])
    with pytest.raises(pa.errors.SchemaError):
        MACRO_SCHEMA.validate(bad)
