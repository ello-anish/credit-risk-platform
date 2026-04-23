"""PD Python evaluation metrics + diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS = sup_t | F_neg(t) - F_pos(t) | over the score distribution."""
    p_pos = y_prob[y_true == 1]
    p_neg = y_prob[y_true == 0]
    if len(p_pos) == 0 or len(p_neg) == 0:
        return float("nan")
    return float(stats.ks_2samp(p_pos, p_neg).statistic)


def gini_coefficient(auc: float) -> float:
    return 2.0 * auc - 1.0


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Return the standard scorecard metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out: dict[str, Any] = {}
    if len(np.unique(y_true)) < 2:
        return {"auc": float("nan"), "ks": float("nan"), "gini": float("nan"),
                "brier": float("nan"), "n": int(len(y_true)),
                "default_rate": float(np.mean(y_true)) if len(y_true) else float("nan")}
    out["auc"] = float(roc_auc_score(y_true, y_prob))
    out["ks"] = ks_statistic(y_true, y_prob)
    out["gini"] = gini_coefficient(out["auc"])
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["n"] = int(len(y_true))
    out["default_rate"] = float(np.mean(y_true))
    return out


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Binned calibration: bucket by decile of predicted prob, compare to actual."""
    edges = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame(columns=["bin", "n", "mean_pred", "mean_actual"])
    edges[0], edges[-1] = -np.inf, np.inf
    binned = pd.cut(y_prob, edges, labels=False, include_lowest=True)
    df = pd.DataFrame({"bin": binned, "y": y_true, "p": y_prob})
    return (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_pred=("p", "mean"), mean_actual=("y", "mean"))
        .reset_index()
    )


def lift_table(y_true: np.ndarray, y_prob: np.ndarray, deciles: int = 10) -> pd.DataFrame:
    """Decile lift: population sorted by descending score, cumulative default capture."""
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, deciles, labels=False, duplicates="drop") + 1
    g = df.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"))
    g["default_rate"] = g["defaults"] / g["n"]
    total_defaults = df["y"].sum()
    g["cumulative_capture"] = g["defaults"].cumsum() / max(total_defaults, 1)
    g["lift"] = g["default_rate"] / df["y"].mean() if df["y"].mean() > 0 else np.nan
    return g.reset_index()
