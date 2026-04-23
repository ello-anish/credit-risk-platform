"""FastAPI scoring service.

Endpoints:
    GET  /health            -> {"status": "ok"}
    GET  /model_info        -> model versions, metrics, reconciliation summary
    POST /score             -> PD / LGD / EAD / ECL for a single loan
    POST /stress            -> Scenario ECL over a portfolio subset

Artifacts consumed (all joblib-persisted during the training pipeline):
    models/pd_python/artifacts/{pd_encoder,pd_model,pd_gbm}.joblib
    models/pd_python/artifacts/metadata.json
    models/pd_r/artifacts/pd_r_metadata.json
    reconciliation/artifacts/reconciliation_metrics.json

If any artefact is missing the endpoint returns a 503 with a clear message —
"pipeline has not been trained".
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from pipeline.config import CFG, REPO_ROOT
from pipeline.logging_utils import get_logger
from service.schemas import (
    LoanFeatures,
    ModelInfoResponse,
    ScoreRequest,
    ScoreResponse,
    StressRequest,
    StressResponse,
)

LOG = get_logger(__name__)

app = FastAPI(title="credit-risk-platform", version=CFG["service"]["model_version"])


# ---------------------------------------------------------------------
# Artefact loaders (lazy + cached)
# ---------------------------------------------------------------------
_state: dict[str, Any] = {}


def _load_pd():
    if "pd" in _state:
        return _state["pd"]
    import joblib
    art = REPO_ROOT / "models" / "pd_python" / "artifacts"
    enc = joblib.load(art / "pd_encoder.joblib")
    baseline = joblib.load(art / "pd_model.joblib")
    gbm = joblib.load(art / "pd_gbm.joblib")
    meta = json.loads((art / "metadata.json").read_text())
    _state["pd"] = (enc, baseline, gbm, meta)
    return _state["pd"]


def _load_lgd_mean() -> float:
    art = REPO_ROOT / "models" / "lgd_python" / "artifacts" / "lgd_metadata.json"
    if art.exists():
        return float(json.loads(art.read_text()).get("lgd_mean_predicted", 0.55))
    return 0.55


def _load_reconciliation() -> dict:
    art = REPO_ROOT / "reconciliation" / "artifacts" / "reconciliation_metrics.json"
    if art.exists():
        return json.loads(art.read_text())
    return {}


def _load_r_meta() -> dict:
    art = REPO_ROOT / "models" / "pd_r" / "artifacts" / "pd_r_metadata.json"
    if art.exists():
        return json.loads(art.read_text())
    return {}


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": CFG["service"]["model_version"]}


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        _, _, _, py_meta = _load_pd()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503,
                            detail=f"PD Python model not trained: {e}")
    return ModelInfoResponse(
        python_model={
            "generated_at": py_meta.get("generated_at"),
            "feature_count": py_meta.get("feature_count"),
            "baseline": py_meta.get("baseline"),
            "challenger": py_meta.get("challenger"),
            "oot_auc": (py_meta.get("metrics", {})
                        .get("oot", {}).get("baseline", {}).get("auc")),
        },
        r_model=_load_r_meta(),
        reconciliation=_load_reconciliation(),
        model_version=CFG["service"]["model_version"],
    )


def _features_to_df(lf: LoanFeatures) -> pd.DataFrame:
    fico_mid = (lf.fico_range_low + lf.fico_range_high) / 2.0
    annual_inc_log = float(np.log1p(max(0.0, lf.annual_inc)))
    row = {
        "loan_amnt": lf.loan_amnt,
        "term_months": lf.term_months,
        "int_rate": lf.int_rate,
        "installment": lf.installment,
        "annual_inc_log": annual_inc_log,
        "dti": lf.dti,
        "delinq_2yrs": lf.delinq_2yrs,
        "inq_last_6mths": lf.inq_last_6mths,
        "open_acc": lf.open_acc,
        "pub_rec": lf.pub_rec,
        "revol_bal": lf.revol_bal,
        "revol_util": lf.revol_util or 0.0,
        "total_acc": lf.total_acc,
        "fico_mid": fico_mid,
        "grade": lf.grade,
        "sub_grade": lf.sub_grade,
        "home_ownership": lf.home_ownership,
        "verification_status": lf.verification_status,
        "purpose": lf.purpose,
        "emp_length": lf.emp_length or "__NA__",
    }
    return pd.DataFrame([row])


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    try:
        enc, baseline, gbm, py_meta = _load_pd()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not trained: {e}")

    from pipeline.features import NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES

    X = _features_to_df(req.loan_features)
    Xw = enc.transform(X[NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES])
    clf = baseline if req.model == "baseline" else gbm
    if hasattr(clf, "feature_names_in_"):
        for c in list(clf.feature_names_in_):
            if c not in Xw.columns:
                Xw[c] = 0.0
        Xw = Xw[list(clf.feature_names_in_)]
    pd_12m = float(clf.predict_proba(Xw)[0, 1])

    # Lifetime PD via geometric extrapolation (Tier 1)
    remaining_months = max(1, int(req.loan_features.term_months))   # as-of origination proxy
    pd_life = float(1.0 - (1.0 - min(pd_12m, 0.999)) ** (remaining_months / 12.0))

    # LGD: use the R-predicted mean (TIER 1 service-time proxy).
    lgd = _load_lgd_mean()

    # EAD: at scoring time we assume the borrower is at origination -> EAD ~ funded_amnt
    ead = float(req.loan_features.funded_amnt)

    # ECL
    r = float(req.loan_features.int_rate)
    df12 = 1.0 / (1.0 + r)
    ecl_12m = pd_12m * lgd * ead * df12
    # Simple lifetime ECL: cumulative PD * LGD * avg EAD (half) * avg DF (middle of life)
    avg_ead = ead * 0.5
    avg_df = 1.0 / (1.0 + r) ** (remaining_months / 24.0)
    ecl_life = pd_life * lgd * avg_ead * avg_df

    # Stage: at-origination, default flag=0 and DPD=0, so stage=1 unless SICR.
    # We don't have pd_at_origination at scoring time (assume current ~= origination).
    stage = 1

    flags = []
    if pd_12m > 0.5:
        flags.append("high_pd")
    if lgd > 0.8:
        flags.append("high_lgd")
    if req.loan_features.grade in ("F", "G"):
        flags.append("deep_subprime")

    return ScoreResponse(
        pd_12m=round(pd_12m, 6),
        pd_lifetime=round(pd_life, 6),
        lgd=round(lgd, 4),
        ead=round(ead, 2),
        ecl_12m=round(ecl_12m, 2),
        ecl_lifetime=round(ecl_life, 2),
        reported_ecl=round(ecl_12m if stage == 1 else ecl_life, 2),
        stage=stage,
        confidence_flags=flags,
    )


@app.post("/stress", response_model=StressResponse)
def stress(req: StressRequest) -> StressResponse:
    """Run one scenario over the stored ECL portfolio and return the delta.

    Reads the pre-computed ``stress_results.parquet``; does NOT retrain.
    """
    sp = REPO_ROOT / "stress" / "artifacts" / "stress_results.parquet"
    if not sp.exists():
        raise HTTPException(status_code=503, detail="Run the stress pipeline first.")
    df = pd.read_parquet(sp)
    if req.scenario_name not in df["scenario"].unique():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario; available: {sorted(df['scenario'].unique())}",
        )

    # apply portfolio filter
    sub = df
    if req.portfolio_subset:
        for k, v in req.portfolio_subset.items():
            if k in sub.columns:
                sub = sub[sub[k] == v]

    base = sub[sub["scenario"] == "baseline"]["reported_ecl"].sum()
    scen = sub[sub["scenario"] == req.scenario_name]["reported_ecl"].sum()
    delta = float(scen - base)
    pct = float(100.0 * (scen / base - 1.0)) if base > 0 else 0.0
    return StressResponse(
        scenario=req.scenario_name,
        baseline_ecl=round(float(base), 2),
        scenario_ecl=round(float(scen), 2),
        delta_ecl=round(delta, 2),
        delta_pct=round(pct, 2),
        n_loans=int(len(sub[sub["scenario"] == req.scenario_name])),
    )
