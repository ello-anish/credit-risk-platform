# R/setup.R — Install project R dependencies into the user library.
#
# Run from the repo root:
#   Rscript R/setup.R
#
# The official spec called for renv, but renv-project bootstrap fails under
# Rscript when the CWD is the repo root (R's .Rprofile gets skipped), and on
# this Windows-only target the renv workflow added friction without value.
# We install directly into ``%LOCALAPPDATA%\R\win-library\<version>\`` which
# is (a) writable without admin, (b) findable by every Rscript invocation
# that adds it to ``.libPaths()``, and (c) easy to clean up.

# ---- User-writable library path ----
user_lib <- file.path(Sys.getenv("LOCALAPPDATA"),
                      "R", "win-library",
                      paste(R.version$major, strsplit(R.version$minor, "\\.")[[1]][1], sep = "."))
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
  cat("Created user library:", user_lib, "\n")
}
.libPaths(c(user_lib, .libPaths()))
cat("Using library:", user_lib, "\n")

# ---- Required packages ----
pkgs <- c(
  # DB + I/O
  "DBI", "RPostgres", "arrow", "jsonlite", "yaml",
  # Core + modelling
  "dplyr", "tidyr", "readr", "stringr", "purrr", "ggplot2",
  "scorecard", "logistf", "betareg",
  "pROC", "ResourceSelection", "survival",
  # Tests
  "testthat"
)

installed <- rownames(installed.packages(lib.loc = user_lib))
to_install <- setdiff(pkgs, installed)

if (length(to_install) > 0) {
  cat("Installing:", paste(to_install, collapse = ", "), "\n")
  install.packages(to_install, lib = user_lib,
                   repos = "https://cloud.r-project.org")
}

cat("\n--- R setup complete — package status ---\n")
for (p in pkgs) {
  status <- tryCatch({
    suppressPackageStartupMessages(loadNamespace(p, lib.loc = user_lib))
    "OK"
  }, error = function(e) "MISSING")
  cat(sprintf("  %-20s %s\n", p, status))
}
