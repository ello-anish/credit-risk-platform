# R/lgd_r/02_betareg.R — Fit a beta regression on LGD in [0,1].
#
# Beta regression requires targets strictly in (0,1) — we apply the standard
# Smithson-Verkuilen boundary adjustment:
#   y' = (y * (n-1) + 0.5) / n
# where n is the sample size of the training slice.

suppressPackageStartupMessages({
  library(betareg)
})

smithson_adj <- function(y) {
  n <- length(y)
  (y * (n - 1) + 0.5) / n
}

fit_lgd <- function(train_df, features, target = "lgd") {
  # Drop rows where target or any feature is NA/Inf — betareg::optim is
  # sensitive to non-finite values and bails with "non-finite value supplied".
  keep_cols <- c(features, target)
  clean <- train_df[, keep_cols, drop = FALSE]
  keep <- stats::complete.cases(clean) &
          apply(clean, 1, function(r) all(is.finite(suppressWarnings(as.numeric(r)))
                                           | is.character(r)))
  dropped <- sum(!keep)
  if (dropped > 0) {
    cat(sprintf("[lgd_r] dropped %d rows with NA/Inf features\n", dropped))
  }
  train_df <- train_df[keep, , drop = FALSE]

  y <- train_df[[target]]
  y_adj <- smithson_adj(y)
  train_df$y_adj <- y_adj
  formula_str <- paste("y_adj ~", paste(features, collapse = " + "))
  fmla <- as.formula(formula_str)
  betareg::betareg(fmla, data = train_df, link = "logit")
}

predict_lgd <- function(fit, df, features) {
  X <- df[, features, drop = FALSE]
  # Ensure categorical levels match training — drop rows whose categoricals
  # are unseen to avoid predict() errors.
  p <- tryCatch(predict(fit, newdata = df, type = "response"),
                error = function(e) {
                  message("predict(betareg) failed: ", conditionMessage(e),
                          "\n  falling back to training-mean LGD")
                  rep(mean(df$lgd, na.rm = TRUE), nrow(df))
                })
  # Clip to [0,1] for safety (betareg is (0,1)-bounded but predict can edge-case)
  pmax(0, pmin(1, p))
}
