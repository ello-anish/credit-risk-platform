"""Pydantic request/response schemas for the scoring service."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class LoanFeatures(BaseModel):
    """Origination-time features needed for PD scoring."""
    loan_amnt: float = Field(..., gt=0)
    funded_amnt: float = Field(..., gt=0)
    term_months: int = Field(..., ge=1)
    int_rate: float = Field(..., ge=0.0, le=0.5, description="annual rate as fraction")
    installment: float = Field(..., gt=0)
    grade: str
    sub_grade: str
    emp_length: Optional[str] = None
    home_ownership: str
    annual_inc: float = Field(..., ge=0)
    verification_status: str
    purpose: str
    dti: float = Field(..., ge=-1, le=100)
    delinq_2yrs: int = Field(0, ge=0)
    inq_last_6mths: int = Field(0, ge=0)
    open_acc: int = Field(0, ge=0)
    pub_rec: int = Field(0, ge=0)
    revol_bal: float = Field(0, ge=0)
    revol_util: Optional[float] = Field(None, ge=0, le=5.0)
    total_acc: int = Field(0, ge=0)
    fico_range_low: float = Field(..., ge=300, le=850)
    fico_range_high: float = Field(..., ge=300, le=850)


class ScoreRequest(BaseModel):
    loan_features: LoanFeatures
    as_of_date: date
    model: Literal["baseline", "gbm"] = "baseline"


class ScoreResponse(BaseModel):
    pd_12m: float
    pd_lifetime: float
    lgd: float
    ead: float
    ecl_12m: float
    ecl_lifetime: float
    reported_ecl: float
    stage: int
    confidence_flags: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    python_model: dict
    r_model: dict
    reconciliation: dict
    model_version: str


class StressRequest(BaseModel):
    scenario_name: str
    portfolio_subset: Optional[dict] = None   # e.g. {"grade": "D", "vintage": "2018-Q2"}


class StressResponse(BaseModel):
    scenario: str
    baseline_ecl: float
    scenario_ecl: float
    delta_ecl: float
    delta_pct: float
    n_loans: int
