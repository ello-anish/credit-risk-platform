"""FastAPI smoke tests — health, model_info, score, stress endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pipeline.config import REPO_ROOT
from service.main import app

client = TestClient(app)


def _have_pd_artifacts() -> bool:
    return (REPO_ROOT / "models" / "pd_python" / "artifacts" / "pd_model.joblib").exists()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.skipif(not _have_pd_artifacts(), reason="PD model not trained yet.")
def test_model_info():
    resp = client.get("/model_info")
    assert resp.status_code == 200
    data = resp.json()
    assert "python_model" in data
    assert data["model_version"]


@pytest.mark.skipif(not _have_pd_artifacts(), reason="PD model not trained yet.")
def test_score():
    payload = {
        "loan_features": {
            "loan_amnt": 10000,
            "funded_amnt": 10000,
            "term_months": 36,
            "int_rate": 0.12,
            "installment": 332,
            "grade": "C",
            "sub_grade": "C3",
            "emp_length": "5 years",
            "home_ownership": "RENT",
            "annual_inc": 55000,
            "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "dti": 18.0,
            "delinq_2yrs": 0,
            "inq_last_6mths": 1,
            "open_acc": 10,
            "pub_rec": 0,
            "revol_bal": 5000,
            "revol_util": 0.3,
            "total_acc": 20,
            "fico_range_low": 695,
            "fico_range_high": 699,
        },
        "as_of_date": "2019-01-01",
        "model": "baseline",
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for k in ("pd_12m", "pd_lifetime", "lgd", "ead", "ecl_12m", "reported_ecl", "stage"):
        assert k in body
    assert 0.0 <= body["pd_12m"] <= 1.0


@pytest.mark.skipif(
    not (REPO_ROOT / "stress" / "artifacts" / "stress_results.parquet").exists(),
    reason="stress results missing",
)
def test_stress_adverse():
    resp = client.post("/stress", json={"scenario_name": "adverse"})
    assert resp.status_code == 200
    data = resp.json()
    # baseline_ecl should be > 0 and delta_ecl positive for adverse
    assert data["baseline_ecl"] > 0
    assert data["delta_ecl"] >= 0
