# Stress Scenario Report

Generated: 2026-04-24T03:37:55.158625

## Vasicek link fit

- alpha: -1.2679
- beta_unemp: +0.6589
- beta_gdp:   +0.1620
- beta_hpi:   +0.0017
- R²: 0.439  (n=28 vintage-quarters)

## Portfolio ECL by scenario

```
        scenario  total_ecl  stage1_ecl  stage2_ecl  stage3_ecl  mean_pd  mean_lgd  n_loans  ecl_delta_vs_baseline  ecl_pct_vs_baseline
        baseline 22,898,975  14,221,019           0   8,677,956        0         1    29278                      0                    0
         adverse 47,080,729   9,271,752  21,238,027  16,570,950        1         1    29278             24,181,754                  106
severely_adverse 56,834,754   4,767,509  32,510,107  19,557,138        1         1    29278             33,935,779                  148
india_rate_shock 35,657,426  22,053,868     833,390  12,770,168        0         1    29278             12,758,451                   56
```

See ``stress_results.parquet`` for per-loan outputs.