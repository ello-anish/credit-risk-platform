"""PD Python training — WoE -> LogisticRegression + GBM challenger + isotonic calibration.

Outputs to ``models/pd_python/artifacts/``:

    pd_model.joblib      # calibrated baseline (the production pick)
    pd_gbm.joblib        # challenger (gradient boosting, NOT calibrated)
    pd_encoder.joblib    # fitted WoE encoder
    metadata.json        # features, hyperparams, metrics
    pd_python_predictions.parquet   # for reconciliation step 6

The baseline is the logistic regression calibrated on the validation slice.
Scoring consumes the joblib bundle: (encoder, classifier).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.config import CFG, REPO_ROOT
from pipeline.features import WoEEncoder, NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES
from pipeline.logging_utils import get_logger

LOG = get_logger(__name__)

ARTIFACTS_DIR: Path = REPO_ROOT / "models" / "pd_python" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _encoder_and_matrix(
    loans_plus: pd.DataFrame,
) -> tuple[WoEEncoder, dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Fit WoE on train, transform all three splits."""
    mod = loans_plus[loans_plus["default_flag"].notna()].copy()
    splits = {s: mod[mod["split"] == s].copy() for s in ("train", "validation", "oot")}
    X_train = splits["train"][NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES]
    y_train = splits["train"]["default_flag"].astype(int)

    enc = WoEEncoder(NUMERIC_CANDIDATES, CATEGORICAL_CANDIDATES, n_bins=10)
    enc.fit(X_train, y_train)

    iv_min = CFG["features"]["iv_min"]
    iv_max = CFG["features"]["iv_max"]
    kept = [
        f for f, iv in enc.iv_.items() if iv_min <= iv <= iv_max
    ]
    keep_cols = [f"{f}__woe" for f in kept]

    Xs: dict[str, pd.DataFrame] = {}
    ys: dict[str, pd.Series] = {}
    for s, df in splits.items():
        Xraw = df[NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES]
        Xw = enc.transform(Xraw)[keep_cols]
        Xs[s] = Xw
        ys[s] = df["default_flag"].astype(int)
    return enc, Xs, ys


def _fit_baseline(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    cfg_b = CFG["pd_python"]["baseline"]
    clf = LogisticRegression(
        C=cfg_b["C"],
        penalty=cfg_b["penalty"],
        max_iter=cfg_b["max_iter"],
        class_weight=cfg_b.get("class_weight"),
        solver="lbfgs",
        random_state=CFG["data"]["seed"],
    )
    clf.fit(X, y)
    return clf


def _fit_challenger(X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
    cfg_c = CFG["pd_python"]["challenger"]
    clf = GradientBoostingClassifier(
        n_estimators=cfg_c["n_estimators"],
        max_depth=cfg_c["max_depth"],
        learning_rate=cfg_c["learning_rate"],
        subsample=cfg_c.get("subsample", 1.0),
        random_state=CFG["data"]["seed"],
    )
    clf.fit(X, y)
    return clf


def train_pd(loans_plus: pd.DataFrame) -> dict[str, Any]:
    """Fit both models, calibrate the baseline on validation, persist artefacts.

    Returns a metrics dict ready for the CLI / MLflow logging.
    """
    from models.pd_python.evaluate import evaluate_model

    enc, Xs, ys = _encoder_and_matrix(loans_plus)
    LOG.info("Feature columns used: %d", Xs["train"].shape[1])

    # ---- Baseline ----
    baseline = _fit_baseline(Xs["train"], ys["train"])

    # Calibrate on validation via prefit + isotonic. We use CV-style prefit
    # (single held-out set is validation). CalibratedClassifierCV(cv='prefit')
    # requires a fitted base estimator and calibrates it on the provided X/y.
    cfg_calib = CFG["pd_python"]["calibration"]
    calibrated = CalibratedClassifierCV(baseline, method=cfg_calib["method"], cv="prefit")
    calibrated.fit(Xs["validation"], ys["validation"])

    # ---- Challenger ----
    gbm = _fit_challenger(Xs["train"], ys["train"])

    # ---- Predictions ----
    probs = {}
    for s in ("train", "validation", "oot"):
        probs[s] = {
            "baseline": calibrated.predict_proba(Xs[s])[:, 1],
            "gbm": gbm.predict_proba(Xs[s])[:, 1],
        }

    # ---- Metrics ----
    metrics = {}
    for s in ("train", "validation", "oot"):
        metrics[s] = {
            "baseline": evaluate_model(ys[s].values, probs[s]["baseline"]),
            "gbm": evaluate_model(ys[s].values, probs[s]["gbm"]),
        }

    # PSI of predicted probs, train vs oot
    from pipeline.features import population_stability_index
    psi = population_stability_index(
        pd.Series(probs["train"]["baseline"]),
        pd.Series(probs["oot"]["baseline"]),
    )
    metrics["psi_baseline_train_oot"] = psi

    # ---- Persist ----
    joblib.dump(enc, ARTIFACTS_DIR / "pd_encoder.joblib")
    joblib.dump(calibrated, ARTIFACTS_DIR / "pd_model.joblib")
    joblib.dump(gbm, ARTIFACTS_DIR / "pd_gbm.joblib")

    pred_rows = []
    mod = loans_plus[loans_plus["default_flag"].notna()].copy()
    for s in ("train", "validation", "oot"):
        sub = mod[mod["split"] == s]
        pred_rows.append(pd.DataFrame({
            "loan_id": sub["loan_id"].values,
            "split": s,
            "prob_default": probs[s]["baseline"],
            "prob_default_gbm": probs[s]["gbm"],
            "default_flag": sub["default_flag"].astype(int).values,
        }))
    preds_df = pd.concat(pred_rows, axis=0, ignore_index=True)
    preds_df.to_parquet(ARTIFACTS_DIR / "pd_python_predictions.parquet",
                        compression="snappy")

    meta = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "feature_count": int(Xs["train"].shape[1]),
        "features": list(Xs["train"].columns),
        "iv": {f: float(enc.iv_.get(f.replace("__woe", ""), 0.0))
               for f in Xs["train"].columns},
        "baseline": {
            "model": "LogisticRegression (isotonic calibrated on validation)",
            "C": CFG["pd_python"]["baseline"]["C"],
            "penalty": CFG["pd_python"]["baseline"]["penalty"],
        },
        "challenger": {
            "model": "GradientBoostingClassifier",
            "n_estimators": CFG["pd_python"]["challenger"]["n_estimators"],
            "max_depth": CFG["pd_python"]["challenger"]["max_depth"],
        },
        "metrics": metrics,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    LOG.info("Saved artefacts to %s", ARTIFACTS_DIR)
    return meta


if __name__ == "__main__":
    clean = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    meta = train_pd(clean)
    print(json.dumps({k: meta[k] for k in ("feature_count", "metrics")}, indent=2, default=str))
