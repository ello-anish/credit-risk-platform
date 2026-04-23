"""Run stress scenarios — fit Vasicek, shift PDs per scenario, recompute ECL.

Reads scenarios from ``stress/scenarios.yml`` (NOT hardcoded).

Outputs:
    stress/artifacts/stress_results.parquet   # long frame: scenario x loan
    stress/artifacts/stress_summary.md        # table of portfolio deltas
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ecl.engine import ECLInput, compute_ecl, portfolio_summary
from ecl.run_ecl import _build_ecl_input, _load_inputs
from pipeline.config import CFG, REPO_ROOT
from pipeline.logging_utils import get_logger
from pipeline.macro import get_macro
from stress.vasicek import VasicekFit, fit_vasicek, shift_pd

LOG = get_logger(__name__)

OUT_DIR = REPO_ROOT / "stress" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCENARIOS_PATH = REPO_ROOT / "stress" / "scenarios.yml"


def _load_scenarios() -> list[dict]:
    with SCENARIOS_PATH.open() as fh:
        raw = yaml.safe_load(fh)
    return raw["scenarios"]


def _run_one(df: pd.DataFrame, inp: ECLInput, fit: VasicekFit, scenario: dict) -> pd.DataFrame:
    shocked = shift_pd(inp.pd_12m, fit, scenario["shocks"])
    inp2 = ECLInput(
        loan_id=inp.loan_id,
        as_of_date=inp.as_of_date,
        pd_12m=shocked,
        pd_origination=inp.pd_origination,
        lgd=inp.lgd,
        ead=inp.ead,
        effective_rate=inp.effective_rate,
        remaining_months=inp.remaining_months,
        days_past_due=inp.days_past_due,
        default_flag_at_asof=inp.default_flag_at_asof,
    )
    out = compute_ecl(inp2, scenario=scenario["name"])
    out["vintage"] = df["vintage"].values
    out["grade"] = df["grade"].values
    return out


def _summarise(results: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    baseline_total = None
    for ecl_df in results:
        scenario = ecl_df["scenario"].iloc[0]
        total = float(ecl_df["reported_ecl"].sum())
        by_stage = portfolio_summary(ecl_df).set_index("stage")["reported_ecl"].to_dict()
        rows.append({
            "scenario": scenario,
            "total_ecl": total,
            "stage1_ecl": by_stage.get(1, 0.0),
            "stage2_ecl": by_stage.get(2, 0.0),
            "stage3_ecl": by_stage.get(3, 0.0),
            "mean_pd": float(ecl_df["pd_12m"].mean()),
            "mean_lgd": float(ecl_df["lgd"].mean()),
            "n_loans": int(len(ecl_df)),
        })
        if scenario == "baseline":
            baseline_total = total
    out = pd.DataFrame(rows)
    if baseline_total:
        out["ecl_delta_vs_baseline"] = out["total_ecl"] - baseline_total
        out["ecl_pct_vs_baseline"] = (out["total_ecl"] / baseline_total - 1.0) * 100.0
    return out


def _write_report(summary: pd.DataFrame, fit: VasicekFit) -> None:
    lines = [
        "# Stress Scenario Report",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "## Vasicek link fit",
        "",
        f"- alpha: {fit.alpha:+.4f}",
        f"- beta_unemp: {fit.beta_unemp:+.4f}",
        f"- beta_gdp:   {fit.beta_gdp:+.4f}",
        f"- beta_hpi:   {fit.beta_hpi:+.4f}",
        f"- R²: {fit.r_squared:.3f}  (n={fit.n} vintage-quarters)",
        "",
        "## Portfolio ECL by scenario",
        "",
        "```",
        summary.to_string(index=False, float_format=lambda v: f"{v:,.0f}"),
        "```",
        "",
        "See ``stress_results.parquet`` for per-loan outputs.",
    ]
    (OUT_DIR / "stress_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_all_scenarios(as_of_date: str = "2019-01-01") -> pd.DataFrame:
    df = _load_inputs()
    inp = _build_ecl_input(df, as_of_date=as_of_date)

    loans_clean = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    macro = get_macro()
    fit = fit_vasicek(loans_clean, macro)
    LOG.info("Vasicek fit: %s", fit.as_dict())

    scenarios = _load_scenarios()
    LOG.info("Running %d scenarios: %s",
             len(scenarios), [s["name"] for s in scenarios])

    results = [_run_one(df, inp, fit, s) for s in scenarios]
    long = pd.concat(results, axis=0, ignore_index=True)
    long.to_parquet(OUT_DIR / "stress_results.parquet", compression="snappy")
    summary = _summarise(results)
    summary.to_csv(OUT_DIR / "stress_summary.csv", index=False)
    _write_report(summary, fit)
    return summary


if __name__ == "__main__":
    summary = run_all_scenarios()
    print("\n=== Stress scenario summary ===")
    print(summary.to_string(index=False))
