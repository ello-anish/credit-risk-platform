"""Streamlit analyst dashboard — 5 tabs over Postgres + parquet artefacts.

Run:   streamlit run dashboard/app.py

Reads from Postgres by default; if the DB is unreachable, falls back to the
parquet artefacts on disk (artifacts/ecl/, stress/artifacts/, etc.). This
means the dashboard works even without a running Postgres instance, which is
useful during demo / README screenshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pipeline.config import CFG, REPO_ROOT

st.set_page_config(
    page_title="credit-risk-platform",
    page_icon=":bank:",
    layout="wide",
)


# ---------------------------------------------------------------------
# Data loaders (DB with parquet fallback)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _try_db(sql: str) -> pd.DataFrame | None:
    try:
        from pipeline.db import read_sql
        return read_sql(sql)
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"DB unreachable, using parquet fallback ({e.__class__.__name__}).")
        return None


@st.cache_data(show_spinner=False)
def load_ecl() -> pd.DataFrame:
    df = _try_db("SELECT * FROM ecl_results WHERE scenario = 'baseline'")
    if df is None or df.empty:
        p = REPO_ROOT / "artifacts" / "ecl" / "ecl_baseline.parquet"
        if p.exists():
            df = pd.read_parquet(p)
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_stress() -> pd.DataFrame:
    p = REPO_ROOT / "stress" / "artifacts" / "stress_results.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_reconciliation() -> dict:
    p = REPO_ROOT / "reconciliation" / "artifacts" / "reconciliation_metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_dq() -> pd.DataFrame:
    df = _try_db("SELECT * FROM data_quality_runs ORDER BY run_at DESC LIMIT 100")
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_pd_metadata() -> dict:
    p = REPO_ROOT / "models" / "pd_python" / "artifacts" / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
def tab_overview() -> None:
    st.header("Portfolio ECL overview")
    ecl = load_ecl()
    if ecl.empty:
        st.info("No ECL results yet — run the pipeline.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Loans", f"{len(ecl):,}")
    c2.metric("Reported ECL", f"${ecl['reported_ecl'].sum():,.0f}")
    c3.metric("Mean PD 12m", f"{ecl['pd_12m'].mean()*100:.2f}%")

    st.subheader("ECL by stage")
    by_stage = (ecl.groupby("stage")
                .agg(n=("loan_id", "size"), ecl=("reported_ecl", "sum"))
                .reset_index())
    st.dataframe(by_stage, use_container_width=True)
    st.bar_chart(by_stage.set_index("stage")["ecl"])


def tab_diagnostics() -> None:
    st.header("Model diagnostics")
    meta = load_pd_metadata()
    rec = load_reconciliation()
    if not meta:
        st.info("PD metadata not found — run the PD Python training step.")
        return

    st.subheader("PD Python — metrics")
    metrics = meta.get("metrics", {})
    for split in ("train", "validation", "oot"):
        if split in metrics:
            st.write(f"**{split}**")
            st.json(metrics[split])

    if rec:
        st.subheader("Python / R reconciliation — OOT")
        tol = CFG["reconciliation"]
        st.json(rec)
        st.write(f"AUC diff: **{rec.get('auc_abs_diff', float('nan')):.4f}** "
                 f"(tol ≤ {tol['auc_abs_diff_max']})")
        st.write(f"Spearman: **{rec.get('spearman', float('nan')):.4f}** "
                 f"(tol ≥ {tol['spearman_min']})")

    plot_path = REPO_ROOT / "reconciliation" / "artifacts" / "reconciliation_plot.png"
    if plot_path.exists():
        st.image(str(plot_path))


def tab_stress() -> None:
    st.header("Stress scenarios")
    df = load_stress()
    if df.empty:
        st.info("No stress results — run stress.scenarios.run_all_scenarios().")
        return
    scenario = st.selectbox("Scenario", sorted(df["scenario"].unique()))
    sub = df[df["scenario"] == scenario]
    base = df[df["scenario"] == "baseline"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio ECL", f"${sub['reported_ecl'].sum():,.0f}")
    delta = float(sub["reported_ecl"].sum() - base["reported_ecl"].sum())
    c2.metric("Δ vs baseline", f"${delta:+,.0f}")
    base_total = base["reported_ecl"].sum()
    if base_total > 0:
        c3.metric("Δ %", f"{delta/base_total*100:+.2f}%")

    st.subheader("By stage")
    by_stage = (sub.groupby("stage")
                .agg(n=("loan_id", "size"), ecl=("reported_ecl", "sum"))
                .reset_index())
    st.dataframe(by_stage, use_container_width=True)


def tab_drill() -> None:
    st.header("Loan drill-down")
    ecl = load_ecl()
    if ecl.empty:
        st.info("No ECL results yet.")
        return
    loan_id = st.selectbox("Loan ID", ecl["loan_id"].head(200).tolist())
    row = ecl[ecl["loan_id"] == loan_id].iloc[0]
    st.json(row.to_dict())


def tab_data_quality() -> None:
    st.header("Data quality")
    dq = load_dq()
    if dq.empty:
        st.info("No DQ history yet — run the ingest pipeline.")
        return
    st.dataframe(dq.head(50), use_container_width=True)
    failed = dq[dq["passed"] == False]
    if len(failed):
        st.error(f"{len(failed)} failed DQ checks in the last 100 runs")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
st.title(":bank: credit-risk-platform")
st.caption("IFRS 9 ECL with Python/R dual-track PD, Vasicek stress testing.")

tabs = st.tabs(["Portfolio", "Diagnostics", "Stress", "Drill-down", "Data quality"])
with tabs[0]: tab_overview()
with tabs[1]: tab_diagnostics()
with tabs[2]: tab_stress()
with tabs[3]: tab_drill()
with tabs[4]: tab_data_quality()
