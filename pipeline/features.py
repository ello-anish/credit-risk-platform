"""Feature engineering — Weight of Evidence (WoE) encoder + IV screening.

Scope:
    * Fit WoE bins on the TRAIN slice only (no look-ahead).
    * Score WoE values on train / validation / oot slices.
    * Filter features by IV range from ``config.yml:features`` (iv_min, iv_max).
    * Write a ``features.yml`` manifest at the repo root listing kept features,
      dropped-for-low-IV, and dropped-for-leakage columns with reasons.
    * Materialise the final feature frame into Postgres as
      ``credit_risk.feature_mart`` (a real table, not a view — view would require
      persisting the bin edges in-database, which is overkill for this project).

Leakage policy:
    PD features must use ONLY information available at loan origination.
    Post-origination fields (``recoveries``, ``last_pymnt_d``, ``total_pymnt``,
    ``out_prncp``, etc.) are hardcoded in ``LEAKAGE_COLUMNS`` and appear in
    ``features.yml:dropped_for_leakage`` with a reason. This is what
    ``tests/test_features.py::test_no_lookahead`` asserts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

from pipeline.config import CFG, REPO_ROOT
from pipeline.db import get_engine
from pipeline.logging_utils import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------
# Policy: columns banned from ever becoming PD features (target leakage or
# post-origination information).
# ---------------------------------------------------------------------
LEAKAGE_COLUMNS: dict[str, str] = {
    "recoveries": "post-origination — recovery amount (used in LGD target)",
    "collection_recovery_fee": "post-origination — collection recovery fee",
    "last_pymnt_d": "post-origination — last payment date",
    "last_pymnt_amnt": "post-origination — last payment amount",
    "total_pymnt": "post-origination — cumulative payments",
    "total_pymnt_inv": "post-origination — cumulative payments to investors",
    "total_rec_int": "post-origination — cumulative interest received",
    "total_rec_late_fee": "post-origination — late fee received",
    "total_rec_prncp": "post-origination — cumulative principal received",
    "next_pymnt_d": "post-origination — scheduled next payment",
    "last_credit_pull_d": "post-origination — last credit pull",
    "out_prncp": "post-origination — outstanding principal",
    "out_prncp_inv": "post-origination — outstanding principal (investor)",
    "loan_status": "target derivative — would leak outcome",
    "loan_status_raw": "target derivative",
    "default_flag": "the target itself",
    "lgd": "LGD target (not PD feature)",
    "issue_d": "raw string form of issue_date — use parsed column only",
}

# Candidate feature sets (origination-time only)
NUMERIC_CANDIDATES = [
    "loan_amnt", "term_months", "int_rate", "installment", "annual_inc_log",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "fico_mid",
]
CATEGORICAL_CANDIDATES = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "emp_length",
]


# ---------------------------------------------------------------------
# WoE / IV primitives
# ---------------------------------------------------------------------
def compute_woe_iv(x_binned: pd.Series, y: pd.Series) -> tuple[dict, float]:
    """WoE per bin and total IV (good=0, bad=1).

    Uses 0.5-count smoothing to avoid infinities on empty cells.
    """
    df = pd.DataFrame({"x": x_binned.astype("string").fillna("__NA__"), "y": y})
    df = df.dropna(subset=["y"])
    total_pos = (df["y"] == 1).sum()
    total_neg = (df["y"] == 0).sum()
    if total_pos == 0 or total_neg == 0:
        return {}, 0.0
    woe: dict[str, float] = {}
    iv = 0.0
    for bin_val, sub in df.groupby("x", dropna=False, observed=True):
        pos = max((sub["y"] == 1).sum(), 0.5)
        neg = max((sub["y"] == 0).sum(), 0.5)
        pct_pos = pos / total_pos
        pct_neg = neg / total_neg
        w = float(np.log(pct_neg / pct_pos))
        woe[str(bin_val)] = w
        iv += (pct_neg - pct_pos) * w
    return woe, float(iv)


def _quantile_bins(s: pd.Series, n: int = 10) -> np.ndarray:
    """Edges for pd.cut — includes -inf / +inf guards and dedupes."""
    qs = np.linspace(0, 1, n + 1)
    edges = np.unique(np.nanquantile(s.dropna().values, qs))
    if len(edges) < 3:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Simple quantile-binning WoE transformer.

    Monotonic enforcement is handled on the R side via the ``scorecard``
    package, which is the canonical scorecard-industry tool. Python-side we
    keep it simple: quantile bins + 0.5-smoothed WoE. The reconciliation
    tests then verify that the two tracks produce similar rankings.
    """

    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        n_bins: int = 10,
        min_bin_frac: float = 0.02,
    ):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.n_bins = n_bins
        self.min_bin_frac = min_bin_frac
        # Fitted state
        self.bins_: dict[str, np.ndarray] = {}
        self.woe_: dict[str, dict[str, float]] = {}
        self.iv_: dict[str, float] = {}
        self.collapsed_rare_: dict[str, set] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        y = y.astype(int)
        for feat in self.numeric:
            edges = _quantile_bins(X[feat], self.n_bins)
            binned = pd.cut(X[feat], edges, include_lowest=True, duplicates="drop")
            woe, iv = compute_woe_iv(binned, y)
            self.bins_[feat] = edges
            self.woe_[feat] = woe
            self.iv_[feat] = iv
        for feat in self.categorical:
            x = X[feat].astype("string").fillna("__NA__")
            counts = x.value_counts(normalize=True)
            rare = set(counts[counts < self.min_bin_frac].index)
            x_collapsed = x.where(~x.isin(rare), "__OTHER__")
            woe, iv = compute_woe_iv(x_collapsed, y)
            self.woe_[feat] = woe
            self.iv_[feat] = iv
            self.collapsed_rare_[feat] = rare
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for feat in self.numeric:
            edges = self.bins_[feat]
            binned = pd.cut(X[feat], edges, include_lowest=True, duplicates="drop")
            binned_str = binned.astype("string").fillna("__NA__")
            out[f"{feat}__woe"] = binned_str.map(self.woe_[feat]).astype(float).fillna(0.0)
        for feat in self.categorical:
            x = X[feat].astype("string").fillna("__NA__")
            rare = self.collapsed_rare_.get(feat, set())
            x = x.where(~x.isin(rare), "__OTHER__")
            known = set(self.woe_[feat].keys())
            x = x.where(x.isin(known), "__OTHER__")
            out[f"{feat}__woe"] = x.map(self.woe_[feat]).astype(float).fillna(0.0)
        return out

    def iv_frame(self) -> pd.DataFrame:
        return (
            pd.Series(self.iv_, name="iv")
            .rename_axis("feature")
            .reset_index()
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------
# PSI — for monitoring drift train->oot
# ---------------------------------------------------------------------
def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10
) -> float:
    """Standard PSI computation (expected vs actual)."""
    edges = np.nanquantile(expected.dropna(), np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    e = np.histogram(expected.dropna(), bins=edges)[0] / len(expected.dropna())
    a = np.histogram(actual.dropna(), bins=edges)[0] / len(actual.dropna())
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------
def _features_yml_path() -> Path:
    return REPO_ROOT / "features.yml"


def select_features_by_iv(
    encoder: WoEEncoder,
) -> tuple[list[str], pd.DataFrame, list[dict]]:
    iv_df = encoder.iv_frame()
    iv_min = CFG["features"]["iv_min"]
    iv_max = CFG["features"]["iv_max"]
    kept: list[str] = []
    dropped: list[dict] = []
    for _, r in iv_df.iterrows():
        f, iv = r["feature"], float(r["iv"])
        if iv < iv_min:
            dropped.append({"feature": f, "iv": iv, "reason": f"IV {iv:.3f} < {iv_min}"})
        elif iv > iv_max:
            dropped.append(
                {"feature": f, "iv": iv, "reason": f"IV {iv:.3f} > {iv_max} (likely leakage)"}
            )
        else:
            kept.append(f)
    return kept, iv_df, dropped


def write_features_manifest(
    kept: list[str],
    iv_df: pd.DataFrame,
    dropped_iv: list[dict],
    encoder: WoEEncoder,
) -> Path:
    iv_lookup = iv_df.set_index("feature")["iv"].astype(float).to_dict()
    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "iv_threshold": {
            "iv_min": CFG["features"]["iv_min"],
            "iv_max": CFG["features"]["iv_max"],
        },
        "counts": {
            "numeric_candidates": len(encoder.numeric),
            "categorical_candidates": len(encoder.categorical),
            "kept": len(kept),
            "dropped_for_iv": len(dropped_iv),
            "dropped_for_leakage": len(LEAKAGE_COLUMNS),
        },
        "kept_features": [{"feature": f, "iv": iv_lookup[f]} for f in kept],
        "dropped_for_iv": dropped_iv,
        "dropped_for_leakage": [
            {"column": k, "reason": v} for k, v in LEAKAGE_COLUMNS.items()
        ],
    }
    path = _features_yml_path()
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, allow_unicode=True)
    LOG.info("Wrote feature manifest %s", path)
    return path


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def build_feature_mart(loans_plus: pd.DataFrame) -> tuple[pd.DataFrame, WoEEncoder, pd.DataFrame]:
    """Fit encoder on train; transform train+val+oot; return (mart, encoder, iv)."""
    modelable = loans_plus[loans_plus["default_flag"].notna()].copy()
    train_mask = modelable["split"] == "train"
    if train_mask.sum() < 1000:
        raise RuntimeError(
            f"Only {train_mask.sum()} training rows with labelled target — "
            "check vintage config or default definition."
        )
    y_train = modelable.loc[train_mask, "default_flag"].astype(int)
    X_train = modelable.loc[train_mask, NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES]

    enc = WoEEncoder(NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES, n_bins=10)
    enc.fit(X_train, y_train)

    kept, iv_df, dropped_iv = select_features_by_iv(enc)
    write_features_manifest(kept, iv_df, dropped_iv, enc)

    X_all = modelable[NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES]
    woe_all = enc.transform(X_all)
    woe_kept_cols = [f"{f}__woe" for f in kept]
    woe_kept = woe_all[woe_kept_cols].reset_index(drop=True)

    meta = modelable[
        ["loan_id", "issue_date", "vintage", "split", "default_flag",
         "grade", "sub_grade", "term_months", "funded_amnt", "int_rate"]
    ].reset_index(drop=True)
    mart = pd.concat([meta, woe_kept], axis=1)
    return mart, enc, iv_df


def materialize_feature_mart(df: pd.DataFrame, table_name: str = "feature_mart") -> None:
    schema = CFG["database"]["schema"]
    engine = get_engine()
    df.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
    LOG.info("Materialised %s.%s (%d rows)", schema, table_name, len(df))


if __name__ == "__main__":
    clean = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    mart, enc, iv_df = build_feature_mart(clean)
    print("\n--- Top 10 features by IV ---")
    print(iv_df.head(10).to_string(index=False))
    print(f"\nKept {sum(1 for f in enc.iv_ if CFG['features']['iv_min'] <= enc.iv_[f] <= CFG['features']['iv_max'])} features, materialising to Postgres...")
    materialize_feature_mart(mart)
