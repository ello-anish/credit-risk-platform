# R/tests/test_binning.R — testthat over the R PD binning + scorecard math.
# Run with:  Rscript -e "testthat::test_dir('R/tests')"

suppressPackageStartupMessages({
  library(testthat)
})

this_dir <- dirname(normalizePath(sys.frame(1)$ofile %||% "R/tests", winslash = "/"))
repo_root <- normalizePath(file.path(this_dir, "..", ".."), winslash = "/")
source(file.path(repo_root, "R", "pd_r", "02_binning.R"))
source(file.path(repo_root, "R", "pd_r", "03_scorecard.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---- Tiny synthetic dataset: higher fico_mid => lower default rate ----
set.seed(42)
n <- 500
dfico <- data.frame(
  fico_mid = sample(600:800, n, replace = TRUE),
  grade    = sample(c("A","B","C","D"), n, replace = TRUE)
)
# Default prob decreasing in FICO
p <- plogis(3 - 0.03 * (dfico$fico_mid - 600))
dfico$default_flag <- rbinom(n, 1, p)

test_that("scorecard::woebin returns monotonic WoE for FICO (synthetic)", {
  skip_if_not_installed("scorecard")
  bins <- scorecard::woebin(
    dt = dfico, y = "default_flag", x = "fico_mid",
    method = "tree", bin_num_limit = 5, positive = "1",
    print_info = FALSE
  )
  woe_vec <- bins$fico_mid$woe
  # Either monotonically increasing OR decreasing; no zigzag
  diffs <- diff(woe_vec)
  signs <- sign(diffs[abs(diffs) > 1e-6])
  if (length(signs) >= 2) {
    expect_true(all(signs == signs[1]),
                info = paste("woe not monotonic:", paste(woe_vec, collapse = ",")))
  } else {
    succeed()
  }
})

test_that("iv is positive on the FICO -> default_flag relationship", {
  skip_if_not_installed("scorecard")
  iv_tbl <- scorecard::iv(dfico[, c("fico_mid", "default_flag")], y = "default_flag")
  expect_gt(iv_tbl$info_value[iv_tbl$variable == "fico_mid"], 0.01)
})
