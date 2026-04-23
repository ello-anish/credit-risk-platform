# R/setup.R — Bootstrap renv and install project dependencies.
#
# Run from the repo root:
#   Rscript R/setup.R
#
# On first run this initializes renv in the R/ subdirectory, installs the
# required CRAN packages, and writes renv.lock. Subsequent runs are no-ops
# when all packages are already installed.

# ---- Locate repo root robustly (works under Rscript, source(), or interactive) ----
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/")))
  }
  if (!is.null(sys.frames()) && length(sys.frames()) >= 1 &&
      !is.null(sys.frame(1)$ofile)) {
    return(dirname(normalizePath(sys.frame(1)$ofile, winslash = "/")))
  }
  # Fallback: assume CWD is repo root and this file is R/setup.R
  return(normalizePath("R", winslash = "/", mustWork = FALSE))
}

r_dir <- get_script_dir()
cat("R dir (renv project):", r_dir, "\n")
setwd(r_dir)

# ---- Bootstrap renv itself ----
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

# ---- Init renv if not already initialised ----
if (!file.exists("renv.lock")) {
  cat("Initializing renv (bare, no dependency discovery)...\n")
  renv::init(bare = TRUE, restart = FALSE)
}

# Activate renv for this session
suppressMessages(renv::activate())

# ---- Required packages ----
pkgs <- c(
  # DB + I/O
  "DBI",
  "RPostgres",
  "arrow",
  "jsonlite",
  "yaml",
  # Core + modelling
  "tidyverse",
  "scorecard",
  "logistf",
  "betareg",
  "pROC",
  "ResourceSelection",
  "survival",
  # MLOps + tests
  "mlflow",
  "testthat"
)

installed <- rownames(installed.packages())
to_install <- setdiff(pkgs, installed)

if (length(to_install) > 0) {
  cat("Installing R packages:", paste(to_install, collapse = ", "), "\n")
  install.packages(to_install, repos = "https://cloud.r-project.org")
} else {
  cat("All R packages already installed.\n")
}

# ---- Snapshot lockfile (best-effort) ----
tryCatch({
  renv::snapshot(prompt = FALSE)
  cat("renv.lock updated.\n")
}, error = function(e) {
  cat("renv::snapshot warning:", conditionMessage(e), "\n")
})

# ---- Summary ----
installed_now <- installed.packages()
have <- intersect(pkgs, rownames(installed_now))
cat("\n--- R setup complete ---\n")
if (length(have) > 0) {
  print(installed_now[have, c("Package", "Version"), drop = FALSE])
}
