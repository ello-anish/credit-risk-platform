"""Stress test sanity — adverse > baseline, severely_adverse > adverse."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.config import REPO_ROOT


STRESS_PATH = REPO_ROOT / "stress" / "artifacts" / "stress_results.parquet"


@pytest.fixture(scope="module")
def stress_df() -> pd.DataFrame:
    if not STRESS_PATH.exists():
        pytest.skip("stress_results.parquet missing — run stress pipeline first.")
    return pd.read_parquet(STRESS_PATH)


def _total(df: pd.DataFrame, scenario: str) -> float:
    return float(df[df["scenario"] == scenario]["reported_ecl"].sum())


def test_adverse_worse_than_baseline(stress_df):
    b = _total(stress_df, "baseline")
    a = _total(stress_df, "adverse")
    assert a > b, f"adverse ECL {a:,.0f} not greater than baseline {b:,.0f}"


def test_severely_adverse_worse_than_adverse(stress_df):
    a = _total(stress_df, "adverse")
    s = _total(stress_df, "severely_adverse")
    assert s > a, f"severely_adverse {s:,.0f} not > adverse {a:,.0f}"


def test_scenarios_present(stress_df):
    scenarios = set(stress_df["scenario"].unique())
    expected = {"baseline", "adverse", "severely_adverse"}
    missing = expected - scenarios
    assert not missing, f"Missing scenarios: {missing}"
