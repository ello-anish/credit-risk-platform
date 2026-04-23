"""Train / validation / OOT split logic, keyed off loan vintage.

The vintage is the issue-date quarter (e.g. "2014-Q2"). Ranges are driven by
``config.yml:vintages``. A loan whose vintage falls outside every defined
range — including pre-2012 ones — is labelled ``excluded``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from pipeline.config import CFG


def issue_date_to_vintage(issue_date: pd.Series) -> pd.Series:
    """Return a vintage label like ``2014-Q2`` per issue_date."""
    issue_date = pd.to_datetime(issue_date)
    return issue_date.dt.year.astype(str) + "-Q" + issue_date.dt.quarter.astype(str)


def _between(s: pd.Series, start: str, end: str) -> pd.Series:
    return (s >= pd.Timestamp(start)) & (s <= pd.Timestamp(end))


def assign_split(issue_date: pd.Series) -> pd.Series:
    """Return a pd.Series of split labels: train / validation / oot / excluded."""
    v = CFG["vintages"]
    d = pd.to_datetime(issue_date)
    split = pd.Series("excluded", index=d.index, dtype="string")
    split.loc[_between(d, v["train_start"], v["train_end"])] = "train"
    split.loc[_between(d, v["validation_start"], v["validation_end"])] = "validation"
    split.loc[_between(d, v["oot_start"], v["oot_end"])] = "oot"
    return split


def split_counts(df: pd.DataFrame, issue_col: str = "issue_date") -> pd.DataFrame:
    """Summary table: rows per split."""
    s = assign_split(df[issue_col])
    return s.value_counts().rename_axis("split").reset_index(name="rows")
