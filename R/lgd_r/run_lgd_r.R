# R/lgd_r/run_lgd_r.R — Entry point for the R LGD track.
#
#   Rscript R/lgd_r/run_lgd_r.R

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  if (length(fa) > 0) return(dirname(normalizePath(sub("^--file=", "", fa[1]), winslash = "/")))
  return(getwd())
}
this_dir <- get_script_dir()
repo_root <- normalizePath(file.path(this_dir, "..", ".."), winslash = "/")

# Ensure the user library is on the path (packages installed by R/setup.R land here).
user_lib <- file.path(Sys.getenv("LOCALAPPDATA"), "R", "win-library", "4.5")
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))

suppressPackageStartupMessages({
  library(arrow)
  library(jsonlite)
  library(dplyr)
})

source(file.path(this_dir, "01_prepare.R"))
source(file.path(this_dir, "02_betareg.R"))

cat("[lgd_r] reading defaults frame from Postgres...\n")
df <- read_lgd_frame()
cat(sprintf("[lgd_r] %d defaulted loans with LGD\n", nrow(df)))

# Minimal well-conditioned feature set for betareg.
# LGD is heavily concentrated at 1.0 (~60% of defaults) on LendingClub, so
# we prefer a small, near-orthogonal feature set over a richer one that
# trips optim convergence. sub_grade is dropped (redundant with grade);
# purpose / home_ownership are dropped (many rare levels -> sparse beta
# log-likelihood). See models/lgd_python/artifacts/lgd_metadata.json for
# the fitted model's summary.
features <- c("grade", "term_months", "int_rate",
              "fico_mid", "annual_inc_log", "dti")

train <- df[df$split %in% c("train", "validation"), ]
oot   <- df[df$split == "oot", ]
if (nrow(train) == 0) {
  # Fall back to using all rows as train if no split-aware defaults are present
  # (can happen with a very small 50k sample where oot defaults are few).
  train <- df
  oot <- df[0, ]
}

cat(sprintf("[lgd_r] train: %d / oot: %d\n", nrow(train), nrow(oot)))

cat("[lgd_r] fitting beta regression (logit link, boundary-adjusted target)...\n")
fit <- fit_lgd(train, features, target = "lgd")
cat("[lgd_r] beta regression summary:\n")
print(summary(fit))

cat("[lgd_r] predicting on train + oot (+ on all defaults for scoring)...\n")
train_p <- predict_lgd(fit, train, features)
oot_p   <- if (nrow(oot) > 0) predict_lgd(fit, oot, features) else numeric(0)
all_p   <- predict_lgd(fit, df, features)

train_rmse <- sqrt(mean((train_p - train$lgd)^2))
oot_rmse <- if (length(oot_p)) sqrt(mean((oot_p - oot$lgd)^2, na.rm = TRUE)) else NA_real_
cat(sprintf("[lgd_r] train RMSE: %.4f  oot RMSE: %.4f\n", train_rmse, oot_rmse))
cat(sprintf("[lgd_r] predicted LGD distribution — min: %.3f  mean: %.3f  max: %.3f\n",
            min(all_p, na.rm = TRUE), mean(all_p, na.rm = TRUE), max(all_p, na.rm = TRUE)))

# -------- export --------
art_dir <- file.path(repo_root, "models", "lgd_python", "artifacts")
dir.create(art_dir, recursive = TRUE, showWarnings = FALSE)

out <- data.frame(
  loan_id     = as.integer(df$loan_id),
  split       = as.character(df$split),
  lgd_actual  = as.numeric(df$lgd),
  lgd_pred    = as.numeric(all_p)
)
# Fill the tiny number of rows where predict_lgd returned NA (due to NA
# features) with the training-mean prediction. Keeps coverage at 100 %.
na_pred <- is.na(out$lgd_pred)
if (any(na_pred)) {
  fill <- mean(all_p, na.rm = TRUE)
  out$lgd_pred[na_pred] <- fill
  cat("[lgd_r] filled", sum(na_pred), "NA predictions with mean",
      round(fill, 4), "\n")
}
pred_path <- file.path(art_dir, "lgd_predictions.parquet")
arrow::write_parquet(out, pred_path, compression = "snappy")
cat("[lgd_r] wrote", pred_path, "(", nrow(out), "rows)\n")

meta <- list(
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  features = features,
  coef_mean = lapply(as.list(coef(fit, model = "mean")),
                      function(x) round(as.numeric(x), 6)),
  coef_precision = lapply(as.list(coef(fit, model = "precision")),
                           function(x) round(as.numeric(x), 6)),
  n_train = nrow(train),
  n_oot = nrow(oot),
  train_rmse = if (is.na(train_rmse)) NA_real_ else round(train_rmse, 4),
  oot_rmse = if (is.na(oot_rmse)) NA_real_ else round(oot_rmse, 4),
  lgd_mean_actual = round(mean(df$lgd, na.rm = TRUE), 4),
  lgd_mean_predicted = round(mean(all_p, na.rm = TRUE), 4)
)
meta_path <- file.path(art_dir, "lgd_metadata.json")
jsonlite::write_json(meta, meta_path, pretty = TRUE, auto_unbox = TRUE,
                      digits = 6, null = "null")
cat("[lgd_r] wrote", meta_path, "\n")
cat("[lgd_r] done.\n")
