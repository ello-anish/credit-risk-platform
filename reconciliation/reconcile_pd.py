"""Python / R PD reconciliation — the dual-language validation gate.

Reads two prediction parquet files (OOT slice only):

    models/pd_python/artifacts/pd_python_predictions.parquet
    models/pd_r/artifacts/pd_r_predictions.parquet

Computes:
    * AUC per track
    * Spearman rank correlation of predicted probabilities
    * Mean absolute probability difference
    * KS between the two probability distributions
    * Agreement at decision thresholds (flag rates at 5 %, 10 %, 20 %)

Writes:
    reconciliation/artifacts/reconciliation_report.md
    reconciliation/artifacts/reconciliation_plot.png  (2x2 panel)

The ``tests/test_reconciliation.py`` gate reads the JSON sidecar and asserts
each metric against config.yml:reconciliation thresholds. Breach = failing
CI build.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from pipeline.config import CFG, REPO_ROOT
from pipeline.logging_utils import get_logger

LOG = get_logger(__name__)

PY_PRED = REPO_ROOT / "models" / "pd_python" / "artifacts" / "pd_python_predictions.parquet"
R_PRED = REPO_ROOT / "models" / "pd_r" / "artifacts" / "pd_r_predictions.parquet"
OUT_DIR = REPO_ROOT / "reconciliation" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_oot_join() -> pd.DataFrame:
    py = pd.read_parquet(PY_PRED)
    r = pd.read_parquet(R_PRED)
    py_oot = py[py["split"] == "oot"][["loan_id", "prob_default", "default_flag"]]
    r_oot = r[r["split"] == "oot"][["loan_id", "prob_default"]].rename(
        columns={"prob_default": "prob_default_r"}
    )
    merged = py_oot.merge(r_oot, on="loan_id", how="inner")
    LOG.info("OOT join: python %d, R %d, joined %d",
             len(py_oot), len(r_oot), len(merged))
    return merged.rename(columns={"prob_default": "prob_default_py"})


def compute_metrics(df: pd.DataFrame) -> dict:
    """Return the full reconciliation metric dict."""
    y = df["default_flag"].astype(int).values
    py = df["prob_default_py"].astype(float).values
    rr = df["prob_default_r"].astype(float).values

    auc_py = float(roc_auc_score(y, py)) if len(np.unique(y)) == 2 else float("nan")
    auc_r = float(roc_auc_score(y, rr)) if len(np.unique(y)) == 2 else float("nan")

    spearman = float(stats.spearmanr(py, rr, nan_policy="omit").statistic)
    mean_abs_diff = float(np.mean(np.abs(py - rr)))
    ks_dist = float(stats.ks_2samp(py, rr).statistic)

    thresholds = {}
    for q in (0.05, 0.10, 0.20):
        # Flag the top-q of each distribution and compute agreement
        cutoff_py = float(np.quantile(py, 1.0 - q))
        cutoff_r = float(np.quantile(rr, 1.0 - q))
        flag_py = (py >= cutoff_py).astype(int)
        flag_r = (rr >= cutoff_r).astype(int)
        agree = float(np.mean(flag_py == flag_r))
        thresholds[f"agreement@top_{int(q*100)}pct"] = agree

    return {
        "n": int(len(df)),
        "auc_py": auc_py,
        "auc_r": auc_r,
        "auc_abs_diff": abs(auc_py - auc_r),
        "spearman": spearman,
        "mean_abs_prob_diff": mean_abs_diff,
        "ks_distribution": ks_dist,
        **thresholds,
    }


def _plot(df: pd.DataFrame, metrics: dict, out_path: Path) -> None:
    y = df["default_flag"].astype(int).values
    py = df["prob_default_py"].values
    rr = df["prob_default_r"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (0,0) ROC overlay
    fpr_py, tpr_py, _ = roc_curve(y, py)
    fpr_r, tpr_r, _ = roc_curve(y, rr)
    ax = axes[0, 0]
    ax.plot(fpr_py, tpr_py, label=f"Python AUC={metrics['auc_py']:.3f}")
    ax.plot(fpr_r, tpr_r, label=f"R AUC={metrics['auc_r']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_title("ROC — OOT")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()

    # (0,1) Calibration (binned mean predicted vs mean actual)
    ax = axes[0, 1]
    for label, p in [("Python", py), ("R", rr)]:
        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
        tbl = (pd.DataFrame({"p": p, "y": y, "bin": dec})
               .groupby("bin").agg(mp=("p", "mean"), my=("y", "mean")).reset_index())
        ax.plot(tbl["mp"], tbl["my"], marker="o", label=label)
    lim = max(py.max(), rr.max(), y.mean() * 2)
    ax.plot([0, lim], [0, lim], "k--", lw=0.5)
    ax.set_title("Calibration — deciles")
    ax.set_xlabel("Mean predicted"); ax.set_ylabel("Mean actual"); ax.legend()

    # (1,0) P-P plot of prob distributions
    ax = axes[1, 0]
    qq = np.linspace(0.01, 0.99, 99)
    ax.plot(np.quantile(py, qq), np.quantile(rr, qq), marker="o", markersize=3)
    lim = max(py.max(), rr.max())
    ax.plot([0, lim], [0, lim], "k--", lw=0.5)
    ax.set_title(f"P-P plot (KS={metrics['ks_distribution']:.3f})")
    ax.set_xlabel("Python quantile"); ax.set_ylabel("R quantile")

    # (1,1) Disagreement histogram
    ax = axes[1, 1]
    diff = py - rr
    ax.hist(diff, bins=50, color="steelblue", alpha=0.8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"Python - R  (mean|diff|={metrics['mean_abs_prob_diff']:.3f})")
    ax.set_xlabel("Python prob - R prob"); ax.set_ylabel("Loans")

    fig.suptitle("PD Reconciliation — OOT slice", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _report(metrics: dict, out_path: Path) -> None:
    tol = CFG["reconciliation"]
    rows = [
        ("AUC (Python)", f"{metrics['auc_py']:.4f}", ""),
        ("AUC (R)", f"{metrics['auc_r']:.4f}", ""),
        ("|AUC_py - AUC_r|", f"{metrics['auc_abs_diff']:.4f}",
         f"<= {tol['auc_abs_diff_max']}"),
        ("Spearman corr(py, r)", f"{metrics['spearman']:.4f}",
         f">= {tol['spearman_min']}"),
        ("Mean |prob_py - prob_r|", f"{metrics['mean_abs_prob_diff']:.4f}",
         f"<= {tol['mean_abs_prob_diff_max']}"),
        ("KS between distributions", f"{metrics['ks_distribution']:.4f}",
         f"<= {tol['ks_distribution_max']}"),
        ("Agreement @ top 5%", f"{metrics['agreement@top_5pct']:.3f}", ""),
        ("Agreement @ top 10%", f"{metrics['agreement@top_10pct']:.3f}", ""),
        ("Agreement @ top 20%", f"{metrics['agreement@top_20pct']:.3f}", ""),
        ("OOT rows reconciled", str(metrics["n"]), ""),
    ]
    lines = [
        "# PD Python / R Reconciliation Report",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "| Metric | Value | Tolerance |",
        "|---|---|---|",
    ]
    for name, val, tol_col in rows:
        lines.append(f"| {name} | {val} | {tol_col} |")
    lines.append("")
    lines.append(
        "See `reconciliation_plot.png` for ROC overlay, calibration, P-P plot, "
        "and disagreement histogram."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def reconcile() -> dict:
    """Run the full reconciliation; writes report + plot + JSON sidecar."""
    df = load_oot_join()
    metrics = compute_metrics(df)
    _plot(df, metrics, OUT_DIR / "reconciliation_plot.png")
    _report(metrics, OUT_DIR / "reconciliation_report.md")
    (OUT_DIR / "reconciliation_metrics.json").write_text(json.dumps(metrics, indent=2))
    LOG.info("Reconciliation metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    m = reconcile()
    print(json.dumps(m, indent=2))
