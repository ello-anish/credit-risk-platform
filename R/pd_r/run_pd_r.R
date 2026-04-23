# R/pd_r/run_pd_r.R — Main entry point for the R PD track.
#
#   Rscript R/pd_r/run_pd_r.R
#
# Steps:
#   1. Read modelling frame from Postgres (01_prepare.R)
#   2. Fit monotonic WoE bins on TRAIN, filter by IV (02_binning.R)
#   3. Fit Firth-penalised logit, build scorecard (03_scorecard.R)
#   4. Validate: AUC / KS / HL / PSI on train + OOT (04_validation.R)
#   5. Export predictions + metadata for Python reconciliation (05_export.R)

# Locate ourselves + activate renv (so installed packages resolve)
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  if (length(fa) > 0) return(dirname(normalizePath(sub("^--file=", "", fa[1]), winslash = "/")))
  return(getwd())
}
this_dir <- get_script_dir()
repo_root <- normalizePath(file.path(this_dir, "..", ".."), winslash = "/")
cat("[pd_r] repo root:", repo_root, "\n")

# Ensure the user library is on the path (packages installed by R/setup.R land here).
user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", "4.5")
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))

suppressPackageStartupMessages({
  library(dplyr)
})

# Source modules
source(file.path(this_dir, "01_prepare.R"))
source(file.path(this_dir, "02_binning.R"))
source(file.path(this_dir, "03_scorecard.R"))
source(file.path(this_dir, "04_validation.R"))
source(file.path(this_dir, "05_export.R"))

cat("[pd_r] reading modelling frame from Postgres...\n")
df <- read_modelling_frame()
cat("[pd_r] frame:", nrow(df), "rows,", ncol(df), "cols\n")

train <- df[df$split == "train", ]
val   <- df[df$split == "validation", ]
oot   <- df[df$split == "oot", ]
cat(sprintf("[pd_r] splits — train: %d / validation: %d / oot: %d\n",
            nrow(train), nrow(val), nrow(oot)))

cat("[pd_r] fitting monotonic WoE bins on train...\n")
cfg_pd <- cfg$pd_r
fit_result <- fit_bins(train,
                       y_col  = "default_flag",
                       iv_min = cfg_pd$iv_min,
                       iv_max = cfg_pd$iv_max)
bins <- fit_result$bins
iv_tbl <- fit_result$iv
kept <- fit_result$kept
cat(sprintf("[pd_r] kept %d features by IV (%.2f - %.2f):\n",
            length(kept), cfg_pd$iv_min, cfg_pd$iv_max))
print(head(iv_tbl, 15))

cat("[pd_r] applying bins to train/val/oot...\n")
train_woe <- apply_bins(train, bins)
val_woe   <- apply_bins(val, bins)
oot_woe   <- apply_bins(oot, bins)

# Ensure default_flag survived
train_woe$default_flag <- train$default_flag
val_woe$default_flag   <- val$default_flag
oot_woe$default_flag   <- oot$default_flag

cat("[pd_r] fitting logistic regression (Firth-penalised)...\n")
fit <- fit_logit(train_woe, y_col = "default_flag")
cat("[pd_r] model class:", class(fit)[1], "\n")

cat("[pd_r] scoring splits (raw, then Platt-calibrated on validation)...\n")
train_p_raw <- predict_prob(fit, train_woe)
val_p_raw   <- predict_prob(fit, val_woe)
oot_p_raw   <- predict_prob(fit, oot_woe)

# Platt scaling on validation. Rank-preserving (important for Spearman/AUC
# reconciliation) and puts the R probs on the same scale as Python's
# CalibratedClassifierCV(method='sigmoid').
train_p <- calibrate_platt(train_p_raw, val_p_raw, val$default_flag)
val_p   <- calibrate_platt(val_p_raw,   val_p_raw, val$default_flag)
oot_p   <- calibrate_platt(oot_p_raw,   val_p_raw, val$default_flag)
cat(sprintf("[pd_r] raw oot mean=%.4f -> calibrated oot mean=%.4f  (actual=%.4f)\n",
            mean(oot_p_raw), mean(oot_p), mean(oot$default_flag)))

cat("[pd_r] building scorecard points table...\n")
card <- build_scorecard(fit, bins,
                        pdo = cfg_pd$pdo,
                        base_points = cfg_pd$base_points,
                        base_odds   = cfg_pd$base_odds)
train_scores <- if (!is.null(card)) {
  scorecard::scorecard_ply(train, card, only_total_score = TRUE)$score
} else NULL
val_scores <- if (!is.null(card)) {
  scorecard::scorecard_ply(val, card, only_total_score = TRUE)$score
} else NULL
oot_scores <- if (!is.null(card)) {
  scorecard::scorecard_ply(oot, card, only_total_score = TRUE)$score
} else NULL

cat("[pd_r] validating...\n")
metrics <- validation_report(train$default_flag, train_p,
                             oot$default_flag,   oot_p)
cat(sprintf("[pd_r] train AUC: %.4f  train KS: %.4f\n",
            metrics$train_auc, metrics$train_ks))
cat(sprintf("[pd_r] oot   AUC: %.4f  oot   KS: %.4f\n",
            metrics$oot_auc,   metrics$oot_ks))
cat(sprintf("[pd_r] PSI (train->oot prob): %.4f\n", metrics$psi_train_oot))

# Print a compact summary of scorecard points per feature (rbind fails when
# the Intercept card has a different column set, so stay list-wise).
if (!is.null(card)) {
  cat("\n[pd_r] scorecard points summary:\n")
  for (name in names(card)) {
    tbl <- as.data.frame(card[[name]], stringsAsFactors = FALSE)
    if ("points" %in% names(tbl)) {
      cat(sprintf("  %-22s range [%+d, %+d]  (%d bins)\n",
                  name,
                  as.integer(min(tbl$points, na.rm = TRUE)),
                  as.integer(max(tbl$points, na.rm = TRUE)),
                  nrow(tbl)))
    }
  }
}

cat("[pd_r] exporting predictions + metadata...\n")

loan_ids <- c(train$loan_id, val$loan_id, oot$loan_id)
splits <- c(rep("train", nrow(train)),
            rep("validation", nrow(val)),
            rep("oot", nrow(oot)))
probs <- c(train_p, val_p, oot_p)
# Simple null-coalesce (must be defined BEFORE use below)
`%||%` <- function(a, b) if (is.null(a)) b else a

scores <- c(train_scores %||% rep(NA_real_, nrow(train)),
            val_scores   %||% rep(NA_real_, nrow(val)),
            oot_scores   %||% rep(NA_real_, nrow(oot)))

export_predictions(repo_root, loan_ids, splits, probs, scores)

bins_summary <- lapply(bins, function(b) {
  data.frame(bin = as.character(b$bin),
             woe = round(b$woe, 4),
             count = b$count,
             pct_distr = round(b$count_distr, 4),
             stringsAsFactors = FALSE)
})
export_metadata(repo_root, fit, bins_summary,
                list(train_auc = metrics$train_auc,
                     train_ks  = metrics$train_ks,
                     oot_auc   = metrics$oot_auc,
                     oot_ks    = metrics$oot_ks,
                     hl_train_pvalue = metrics$hl_train$pvalue,
                     psi_train_oot = metrics$psi_train_oot),
                iv_kept = kept)

cat("[pd_r] done.\n")
