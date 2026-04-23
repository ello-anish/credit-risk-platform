# PD Python / R Reconciliation Report

Generated: 2026-04-24T03:32:50.023256

| Metric | Value | Tolerance |
|---|---|---|
| AUC (Python) | 0.6829 |  |
| AUC (R) | 0.6760 |  |
| |AUC_py - AUC_r| | 0.0069 | <= 0.02 |
| Spearman corr(py, r) | 0.9284 | >= 0.9 |
| Mean |prob_py - prob_r| | 0.0389 | <= 0.03 |
| KS between distributions | 0.0206 | <= 0.05 |
| Agreement @ top 5% | 0.954 |  |
| Agreement @ top 10% | 0.931 |  |
| Agreement @ top 20% | 0.893 |  |
| OOT rows reconciled | 5254 |  |

See `reconciliation_plot.png` for ROC overlay, calibration, P-P plot, and disagreement histogram.