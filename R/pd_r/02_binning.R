# R/pd_r/02_binning.R — Monotonic WoE binning via the `scorecard` package.
#
# This is the canonical scorecard-industry binning step. We ONLY fit bins on
# the TRAIN slice (same policy as the Python track — no look-ahead).

suppressPackageStartupMessages({
  library(scorecard)
  library(dplyr)
})

FEATURE_CANDIDATES <- c(
  # numeric
  "loan_amnt", "term_months", "int_rate", "installment",
  "annual_inc", "dti", "fico_mid",
  "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
  "revol_bal", "revol_util", "total_acc",
  # categorical
  "grade", "sub_grade", "home_ownership", "verification_status",
  "purpose", "emp_length"
)

fit_bins <- function(train_df, y_col = "default_flag",
                     iv_min = 0.02, iv_max = 0.5) {
  stopifnot(y_col %in% names(train_df))

  x_cols <- intersect(FEATURE_CANDIDATES, names(train_df))
  X <- train_df[, c(x_cols, y_col), drop = FALSE]

  # scorecard::woebin does monotonic binning by default for numeric vars.
  bins <- scorecard::woebin(
    dt          = X,
    y           = y_col,
    x           = x_cols,
    method      = "tree",
    bin_num_limit = 6,
    count_distr_limit = 0.03,
    positive    = "1",
    print_info  = FALSE
  )

  # IV per feature
  iv_tbl <- scorecard::iv(X, y = y_col)
  iv_tbl <- iv_tbl[order(-iv_tbl$info_value), ]
  # Filter by IV range
  keep <- iv_tbl$variable[iv_tbl$info_value >= iv_min & iv_tbl$info_value <= iv_max]
  list(bins = bins[keep], iv = iv_tbl, kept = keep)
}

apply_bins <- function(df, bins) {
  # scorecard::woebin_ply converts raw columns to their WoE values
  scorecard::woebin_ply(df, bins = bins, to = "woe", print_info = FALSE)
}
