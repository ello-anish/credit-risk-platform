# credit-risk-platform

> **Status: Step 1 (infra) complete. Awaiting LendingClub sample file before data layer runs.**

Production-grade IFRS 9 Expected Credit Loss (ECL) platform with dual-language
PD modelling (Python + R), Vasicek-link macroeconomic stress testing, a
FastAPI scoring service, and a Streamlit analyst dashboard.

The core differentiator is **dual-language implementation**: every major model
is built in both Python and R, and the two tracks are statistically reconciled
as a CI gate. This mirrors how model validation actually works at banks —
Python for production, R as the independent challenger.

Full documentation will be written in Step 13, after all numeric results are
available to quote verbatim.

## Quickstart (will fail until the sample file is in place — see below)

```bash
# 1. Bring up Postgres
docker compose up -d

# 2. Python deps
python -m venv .venv && source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt

# 3. R deps
Rscript R/setup.R

# 4. End-to-end pipeline
python run_pipeline.py
```
