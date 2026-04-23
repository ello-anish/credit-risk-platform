"""End-to-end orchestrator for the credit-risk-platform pipeline.

Usage:
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --fast       # 10 K-loan smoke sample
    python run_pipeline.py --skip-r     # skip the R tracks (CI without R)
    python run_pipeline.py --tier2      # include survival / DeepSurv (future)

Steps (aborts on the first failure; no silent skipping):

    1. Data check      : HALT if data/raw/lending_club_sample_50k.parquet missing
    2. Ingest           : parquet -> Postgres (loans, loan_status, defaults, macro)
    3. Features         : WoE encoder, IV filter, materialise feature_mart
    4. PD Python        : train + calibrate + persist artefacts
    5. PD R             : Rscript R/pd_r/run_pd_r.R
    6. Reconciliation   : compute metrics + run pytest gate
    7. LGD R            : Rscript R/lgd_r/run_lgd_r.R
    8. ECL engine       : per-loan ECL with staging
    9. Stress           : Vasicek link + scenario ECL
   10. README numbers   : emitted summary JSON consumable by Step 13
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pipeline.config import CFG, REPO_ROOT, raw_path
from pipeline.logging_utils import get_logger

LOG = get_logger("run_pipeline")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _banner(msg: str) -> None:
    bar = "=" * 72
    LOG.info("\n%s\n%s\n%s", bar, msg, bar)


def _run_rscript(script: Path) -> int:
    """Run an R script via the configured rscript binary and stream stdout."""
    cmd = [CFG.get("rscript_path", "Rscript"), str(script)]
    LOG.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def _halt_if_parquet_missing() -> None:
    p = raw_path("lending_club_sample")
    if not p.exists():
        print("HALTING: LendingClub sample file not found at", p)
        print("See README 'Colab instructions' to regenerate.")
        sys.exit(2)


# ---------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------
def step_ingest(fast: bool) -> None:
    _banner("Step 2 - Ingest + Quality Gate")
    from pipeline.ingest import ingest_all
    from pipeline.quality import validate_loans, validate_defaults, validate_macro
    import pandas as pd

    counts = ingest_all(truncate=True)
    LOG.info("Ingest row counts: %s", counts)

    clean = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    if fast:
        n = CFG["data"]["fast_sample_size"]
        seed = CFG["data"]["seed"]
        clean = clean.sample(n=min(n, len(clean)), random_state=seed).reset_index(drop=True)
        clean.to_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet",
                         compression="snappy")
        LOG.info("FAST mode: subsampled to %d loans", len(clean))

    # Run the pandera schemas
    loans_for_validation = clean.assign(
        issue_date=pd.to_datetime(clean["issue_date"]),
    )
    validate_loans(loans_for_validation)


def step_features() -> None:
    _banner("Step 3 - Feature engineering")
    import pandas as pd
    from pipeline.features import build_feature_mart, materialize_feature_mart
    clean = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    mart, enc, iv_df = build_feature_mart(clean)

    LOG.info("Top 10 features by IV:")
    LOG.info("\n%s", iv_df.head(10).to_string(index=False))

    materialize_feature_mart(mart)


def step_pd_python() -> None:
    _banner("Step 4 - PD Python (logistic + GBM, isotonic calibration)")
    import pandas as pd
    from models.pd_python.train import train_pd
    clean = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    meta = train_pd(clean)
    m = meta["metrics"]
    LOG.info("Train (baseline): %s", m["train"]["baseline"])
    LOG.info("OOT (baseline):   %s", m["oot"]["baseline"])
    LOG.info("OOT (gbm):        %s", m["oot"]["gbm"])


def step_pd_r() -> None:
    _banner("Step 5 - PD R scorecard")
    rc = _run_rscript(REPO_ROOT / "R" / "pd_r" / "run_pd_r.R")
    if rc != 0:
        raise RuntimeError(f"R PD pipeline failed (exit {rc})")


def step_reconcile() -> None:
    _banner("Step 6 - PD Python / R reconciliation")
    from reconciliation.reconcile_pd import reconcile
    metrics = reconcile()
    LOG.info("Reconciliation: %s", metrics)

    # Run the pytest gate specifically on reconciliation tests
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_reconciliation.py", "-v"],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError("Reconciliation pytest gate FAILED — see output above.")


def step_lgd_r() -> None:
    _banner("Step 7 - LGD R beta regression")
    rc = _run_rscript(REPO_ROOT / "R" / "lgd_r" / "run_lgd_r.R")
    if rc != 0:
        raise RuntimeError(f"R LGD pipeline failed (exit {rc})")


def step_ead() -> None:
    _banner("Step 8 - EAD (outstanding principal)")
    import numpy as np
    import pandas as pd
    from models.ead_python.ead import predict_ead

    clean = pd.read_parquet(REPO_ROOT / "data" / "features" / "loans_clean.parquet")
    as_of = pd.Timestamp("2019-01-01")
    clean["months_elapsed"] = ((as_of - pd.to_datetime(clean["issue_date"])).dt.days
                                / 30.4375).clip(lower=0)
    clean["ead"] = predict_ead(clean, method="annuity")
    defaults = clean[clean["default_flag"] == 1]
    LOG.info("EAD at default (summary):\n%s", defaults["ead"].describe().to_string())


def step_ecl() -> None:
    _banner("Step 9 - ECL engine")
    from ecl.run_ecl import run_ecl
    from ecl.engine import portfolio_summary
    df = run_ecl()
    LOG.info("Portfolio ECL by stage:\n%s",
             portfolio_summary(df).to_string(index=False))


def step_stress() -> None:
    _banner("Step 10 - Stress scenarios")
    from stress.scenarios import run_all_scenarios
    summary = run_all_scenarios()
    LOG.info("Stress summary:\n%s", summary.to_string(index=False))


def step_final_tests() -> None:
    _banner("Step 12 - Full test suite")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=short"],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError("Test suite FAILED")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="10K-loan smoke sample")
    ap.add_argument("--skip-r", action="store_true", help="skip R tracks (CI without R)")
    ap.add_argument("--tier2", action="store_true", help="include survival (Tier 2)")
    ap.add_argument("--only", default=None,
                    help="Comma list of steps to run (ingest,features,pd_py,pd_r,reconcile,lgd,ead,ecl,stress,tests)")
    args = ap.parse_args(argv)

    _banner("Step 1 - Infra already bootstrapped (see README)")
    _halt_if_parquet_missing()

    all_steps = [
        ("ingest",   lambda: step_ingest(args.fast)),
        ("features", step_features),
        ("pd_py",    step_pd_python),
        ("pd_r",     step_pd_r if not args.skip_r else lambda: LOG.info("skip-r: PD R")),
        ("reconcile", step_reconcile if not args.skip_r else lambda: LOG.info("skip-r: reconcile")),
        ("lgd",      step_lgd_r if not args.skip_r else lambda: LOG.info("skip-r: LGD R")),
        ("ead",      step_ead),
        ("ecl",      step_ecl),
        ("stress",   step_stress),
        ("tests",    step_final_tests),
    ]

    wanted = set(s.strip() for s in args.only.split(",")) if args.only else None
    for name, fn in all_steps:
        if wanted and name not in wanted:
            continue
        try:
            fn()
        except Exception as e:   # noqa: BLE001
            LOG.exception("Step '%s' FAILED: %s", name, e)
            return 1

    _banner("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
