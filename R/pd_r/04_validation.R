# R/pd_r/04_validation.R — PSI, KS, AUC, Hosmer-Lemeshow, PSI.

suppressPackageStartupMessages({
  library(pROC)
  library(ResourceSelection)
})

auc_value <- function(y, p) {
  as.numeric(pROC::auc(pROC::roc(response = y, predictor = p,
                                  quiet = TRUE, direction = "<")))
}

ks_value <- function(y, p) {
  # Kolmogorov-Smirnov on the empirical CDFs of P|y=0 and P|y=1
  p0 <- sort(p[y == 0]); p1 <- sort(p[y == 1])
  if (length(p0) == 0 || length(p1) == 0) return(NA_real_)
  as.numeric(ks.test(p0, p1)$statistic)
}

hosmer_lemeshow <- function(y, p, g = 10) {
  res <- tryCatch(ResourceSelection::hoslem.test(y, p, g = g),
                  error = function(e) NULL)
  if (is.null(res)) return(list(stat = NA_real_, pvalue = NA_real_))
  list(stat = as.numeric(res$statistic),
       pvalue = as.numeric(res$p.value))
}

psi_score <- function(expected, actual, bins = 10) {
  # PSI between two numeric distributions
  qs <- quantile(expected, probs = seq(0, 1, length.out = bins + 1),
                 na.rm = TRUE, type = 7)
  qs <- unique(qs)
  if (length(qs) < 3) return(0)
  qs[1] <- -Inf; qs[length(qs)] <- Inf
  e_hist <- table(cut(expected, breaks = qs, include.lowest = TRUE)) / length(expected)
  a_hist <- table(cut(actual, breaks = qs, include.lowest = TRUE)) / length(actual)
  e_hist[e_hist == 0] <- 1e-6
  a_hist[a_hist == 0] <- 1e-6
  sum((a_hist - e_hist) * log(a_hist / e_hist))
}

validation_report <- function(train_y, train_p, oot_y, oot_p) {
  list(
    train_auc = auc_value(train_y, train_p),
    train_ks  = ks_value(train_y, train_p),
    oot_auc   = auc_value(oot_y, oot_p),
    oot_ks    = ks_value(oot_y, oot_p),
    hl_train  = hosmer_lemeshow(train_y, train_p),
    psi_train_oot = psi_score(train_p, oot_p)
  )
}
