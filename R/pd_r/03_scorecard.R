# R/pd_r/03_scorecard.R — Fit a penalized logistic + build scorecard.
#
# Firth-penalised logit via `logistf` when possible (stable with
# quasi-separation); falls back to standard `glm(binomial)` on numerical errors.

suppressPackageStartupMessages({
  library(scorecard)
  library(logistf)
})

fit_logit <- function(train_woe, y_col = "default_flag") {
  train_woe <- as.data.frame(train_woe, stringsAsFactors = FALSE)
  # woe-encoded feature cols end with "_woe"
  feat_cols <- grep("_woe$", names(train_woe), value = TRUE)
  if (length(feat_cols) == 0) {
    stop("No WoE columns in training frame — did woebin_ply run?")
  }

  formula_str <- paste(y_col, "~", paste(feat_cols, collapse = " + "))
  fmla <- as.formula(formula_str)

  fit <- tryCatch(
    logistf::logistf(fmla, data = train_woe, pl = FALSE),
    error = function(e) {
      message("logistf failed (", conditionMessage(e), "). Falling back to glm().")
      glm(fmla, data = train_woe, family = binomial())
    }
  )
  fit
}

predict_prob <- function(fit, df_woe) {
  # scorecard::woebin_ply returns a data.table; coerce so base `[` selection works
  df_woe <- as.data.frame(df_woe, stringsAsFactors = FALSE)
  feat_cols <- grep("_woe$", names(df_woe), value = TRUE)
  X <- df_woe[, feat_cols, drop = FALSE]
  if (inherits(fit, "logistf")) {
    coefs <- coef(fit)
    intercept <- coefs[["(Intercept)"]]
    beta <- coefs[feat_cols]
    linpred <- intercept + as.matrix(X) %*% beta
    plogis(as.numeric(linpred))
  } else {
    as.numeric(predict(fit, newdata = df_woe, type = "response"))
  }
}

# Platt scaling (a.k.a. sigmoid calibration) — a monotonic, rank-preserving
# calibration that puts the R probabilities on the same footing as the Python
# baseline (which uses sklearn's CalibratedClassifierCV(method='sigmoid')).
# Fits y ~ logit(raw) by logistic regression on the validation slice, then
# applies the fitted sigmoid to any new set of raw probabilities.
calibrate_platt <- function(x_new, x_cal, y_cal) {
  stopifnot(length(x_cal) == length(y_cal), length(x_cal) > 10)
  eps <- 1e-6
  lx_cal <- stats::qlogis(pmin(pmax(x_cal, eps), 1 - eps))
  fit <- suppressWarnings(stats::glm(y ~ lx,
                                      family = stats::binomial(),
                                      data = data.frame(y = y_cal, lx = lx_cal)))
  lx_new <- stats::qlogis(pmin(pmax(x_new, eps), 1 - eps))
  pred <- stats::predict(fit,
                          newdata = data.frame(lx = lx_new),
                          type = "response")
  pmin(pmax(as.numeric(pred), 0.0), 1.0)
}

build_scorecard <- function(fit, bins, pdo = 20, base_points = 600, base_odds = 50) {
  # scorecard::scorecard expects an lm-style fit; logistf mostly works
  # when coerced. If it fails we skip the points card — probabilities are
  # what downstream ECL needs anyway.
  card <- tryCatch(
    scorecard::scorecard(
      bins = bins,
      model = fit,
      points0 = base_points,
      odds0 = 1 / base_odds,
      pdo = pdo
    ),
    error = function(e) {
      message("scorecard::scorecard failed: ", conditionMessage(e))
      NULL
    }
  )
  card
}
