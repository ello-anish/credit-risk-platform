"""The reconciliation gate — Python and R PD tracks must agree on OOT.

Reads reconciliation/artifacts/reconciliation_metrics.json produced by
``reconciliation.reconcile_pd.reconcile()`` and asserts each metric against
the tolerances in config.yml:reconciliation. Breach = failing CI build.

If the metrics file does not exist, the test is skipped (pipeline hasn't
run yet). Skip is intentional — pytest without fixtures is still meaningful
for the other test modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.config import CFG, REPO_ROOT

METRICS_PATH = REPO_ROOT / "reconciliation" / "artifacts" / "reconciliation_metrics.json"


@pytest.fixture(scope="module")
def metrics() -> dict:
    if not METRICS_PATH.exists():
        pytest.skip("reconciliation_metrics.json missing — run pipeline first.")
    return json.loads(METRICS_PATH.read_text())


def test_auc_difference_within_tolerance(metrics):
    tol = CFG["reconciliation"]["auc_abs_diff_max"]
    assert metrics["auc_abs_diff"] <= tol, (
        f"|AUC_py - AUC_r| = {metrics['auc_abs_diff']:.4f} > {tol}"
    )


def test_spearman_above_threshold(metrics):
    tol = CFG["reconciliation"]["spearman_min"]
    assert metrics["spearman"] >= tol, (
        f"Spearman = {metrics['spearman']:.4f} < {tol}"
    )


def test_mean_abs_prob_diff(metrics):
    tol = CFG["reconciliation"]["mean_abs_prob_diff_max"]
    assert metrics["mean_abs_prob_diff"] <= tol, (
        f"mean |diff| = {metrics['mean_abs_prob_diff']:.4f} > {tol}"
    )


def test_ks_between_distributions(metrics):
    tol = CFG["reconciliation"]["ks_distribution_max"]
    assert metrics["ks_distribution"] <= tol, (
        f"KS = {metrics['ks_distribution']:.4f} > {tol}"
    )


def test_at_least_1000_oot_rows(metrics):
    assert metrics["n"] >= 1000, f"Only {metrics['n']} OOT rows reconciled"
