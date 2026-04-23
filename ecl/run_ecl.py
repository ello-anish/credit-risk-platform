"""Assemble the ECL run end-to-end and write results to Postgres + parquet.

Reads:
    * Python PD predictions:    models/pd_python/artifacts/pd_python_predictions.parquet
    * R LGD predictions:         models/lgd_python/artifacts/lgd_predictions.parquet
    * Cleaned loans parquet:     data/features/loans_clean.parquet

Writes:
    * Postgres credit_risk.ecl_results (upserted for scenario='baseline')
    * artifacts/ecl/ecl_baseline.parquet
    * stdout: portfolio summary table
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecl.engine import ECLInput, compute_ecl, portfolio_summary, portfolio_by
from pipeline.config import CFG, REPO_ROOT
from pipeline.db import copy_dataframe, get_engine, set_schema, wait_for_db
from pipeline.logging_utils import get_logger
from sqlalchemy import text

LOG = get_logger(__name__)


def _load_inputs() -> pd.DataFrame:
    """Join PD, LGD, loan characteristics into a single per-loan frame."""
    loans = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    pd_py = pd.read_parquet(
        REPO_ROOT / "models" / "pd_python" / "artifacts" / "pd_python_predictions.parquet"
    )

    lgd_path = REPO_ROOT / "models" / "lgd_python" / "artifacts" / "lgd_predictions.parquet"
    if lgd_path.exists():
        lgd = pd.read_parquet(lgd_path)[["loan_id", "lgd_pred"]]
    else:
        LOG.warning("LGD predictions missing — falling back to global mean LGD 0.55")
        lgd = pd.DataFrame({"loan_id": loans["loan_id"],
                            "lgd_pred": [0.55] * len(loans)})

    # Compose portfolio — one row per loan (OOT + validation) with predicted PD
    # and predicted LGD (LGD is only trained on defaulted loans; for non-
    # defaulted loans we impute the predicted mean from the model).
    df = loans[[
        "loan_id", "issue_date", "term_months", "int_rate",
        "funded_amnt", "default_flag", "split", "grade",
        "vintage",
    ]].copy()

    df = df.merge(pd_py[["loan_id", "prob_default"]], on="loan_id", how="inner")
    df = df.merge(lgd, on="loan_id", how="left")
    # Fill missing LGD with the predicted mean across defaulted loans
    mean_lgd = float(lgd["lgd_pred"].mean())
    df["lgd_pred"] = df["lgd_pred"].fillna(mean_lgd).clip(0.0, 1.0)
    return df


def _build_ecl_input(df: pd.DataFrame, as_of_date: str = "2019-01-01") -> ECLInput:
    """Project each loan forward to ``as_of_date`` and compute required fields."""
    as_of = pd.Timestamp(as_of_date)
    df = df.copy()
    # pandas 2.2 dropped timedelta64('M'); approximate month from 30.4375 days.
    df["months_elapsed"] = ((as_of - pd.to_datetime(df["issue_date"])).dt.days
                            / 30.4375).astype(float).clip(lower=0)
    df["months_elapsed"] = df["months_elapsed"].clip(upper=df["term_months"].astype(float))
    df["remaining_months"] = (df["term_months"].astype(float) - df["months_elapsed"]).clip(lower=1).astype(int)

    # EAD = outstanding principal (annuity amortisation)
    from models.ead_python.ead import predict_ead
    df["ead"] = predict_ead(df, months_elapsed_col="months_elapsed", method="annuity")

    # PD at origination — we use the model-predicted PD_12m as-if-origination
    # since we do not retain per-loan snapshots in Tier 1. For the SICR trigger
    # we therefore rely primarily on the DPD gate. Tier 2 will replace this.
    df["pd_origination"] = df["prob_default"].astype(float)
    # Use 0 DPD / default=0 for non-defaulted; 120/1 for defaulted (as of
    # observation point).
    df["days_past_due"] = np.where(df["default_flag"].astype(int) == 1, 120, 0)
    df["default_flag_at_asof"] = df["default_flag"].astype(int)

    return ECLInput(
        loan_id=df["loan_id"].values,
        as_of_date=np.array([as_of.date()] * len(df)),
        pd_12m=df["prob_default"].astype(float).values,
        pd_origination=df["pd_origination"].values,
        lgd=df["lgd_pred"].astype(float).values,
        ead=df["ead"].astype(float).values,
        effective_rate=df["int_rate"].astype(float).values,
        remaining_months=df["remaining_months"].astype(int).values,
        days_past_due=df["days_past_due"].astype(int).values,
        default_flag_at_asof=df["default_flag_at_asof"].astype(int).values,
    )


def _persist(ecl_df: pd.DataFrame, scenario: str = "baseline") -> None:
    """Upsert into credit_risk.ecl_results for the given scenario."""
    wait_for_db()
    set_schema()
    schema = CFG["database"]["schema"]
    with get_engine().begin() as cx:
        cx.execute(text(f"""
            DELETE FROM {schema}.ecl_results WHERE scenario = :scenario
        """), {"scenario": scenario})

    copy_dataframe(
        ecl_df,
        "ecl_results",
        columns=[
            "loan_id", "as_of_date", "scenario",
            "pd_12m", "pd_lifetime", "lgd", "ead",
            "ecl_12m", "ecl_lifetime", "reported_ecl",
            "stage", "model_version",
        ],
    )


def run_ecl(as_of_date: str = "2019-01-01", scenario: str = "baseline",
            persist: bool = True) -> pd.DataFrame:
    df = _load_inputs()
    inp = _build_ecl_input(df, as_of_date=as_of_date)
    ecl_df = compute_ecl(inp, scenario=scenario)

    out_dir = REPO_ROOT / "artifacts" / "ecl"
    out_dir.mkdir(parents=True, exist_ok=True)
    ecl_df.to_parquet(out_dir / f"ecl_{scenario}.parquet", compression="snappy")

    LOG.info("Portfolio summary (%s):", scenario)
    LOG.info("\n%s", portfolio_summary(ecl_df).to_string(index=False))

    if persist:
        _persist(ecl_df, scenario=scenario)
    return ecl_df


if __name__ == "__main__":
    df = run_ecl()
    print("\n=== Portfolio ECL by stage ===")
    print(portfolio_summary(df).to_string(index=False))
    print("\n=== By grade ===")
    print(portfolio_by(
        df.merge(
            pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
            [["loan_id", "grade"]], on="loan_id"),
        by="grade",
    ).to_string(index=False))
