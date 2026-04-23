# R/pd_r/05_export.R — Export predictions + metadata for Python reconciliation.
#
# Writes:
#   models/pd_r/artifacts/pd_r_predictions.parquet
#     cols: loan_id, split, prob_default, score
#   models/pd_r/artifacts/pd_r_metadata.json
#     coefficients, WoE bins summary, validation metrics

suppressPackageStartupMessages({
  library(arrow)
  library(jsonlite)
})

ARTIFACTS_DIR <- function(repo_root) {
  dir <- file.path(repo_root, "models", "pd_r", "artifacts")
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  dir
}

export_predictions <- function(repo_root, loan_ids, splits, probs, scores) {
  out <- data.frame(
    loan_id      = as.integer(loan_ids),
    split        = as.character(splits),
    prob_default = as.numeric(probs),
    score        = if (is.null(scores)) NA_real_ else as.numeric(scores)
  )
  out_path <- file.path(ARTIFACTS_DIR(repo_root), "pd_r_predictions.parquet")
  arrow::write_parquet(out, out_path, compression = "snappy")
  cat("Wrote", out_path, "(", nrow(out), "rows )\n")
  out_path
}

export_metadata <- function(repo_root, fit, bins_summary, metrics, iv_kept) {
  coef_list <- if (inherits(fit, "logistf")) {
    as.list(coef(fit))
  } else {
    as.list(coefficients(fit))
  }
  # Serialise coefs as named numeric
  coef_list <- lapply(coef_list, function(x) round(as.numeric(x), 6))

  meta <- list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    model_family = if (inherits(fit, "logistf")) "logistf (Firth-penalised)" else "glm(binomial)",
    coefficients = coef_list,
    iv_kept = iv_kept,
    bins_summary = bins_summary,
    metrics = metrics
  )

  out_path <- file.path(ARTIFACTS_DIR(repo_root), "pd_r_metadata.json")
  jsonlite::write_json(meta, out_path, pretty = TRUE, auto_unbox = TRUE,
                        digits = 6, null = "null")
  cat("Wrote", out_path, "\n")
  out_path
}
