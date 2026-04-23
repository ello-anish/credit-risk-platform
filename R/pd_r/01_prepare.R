# R/pd_r/01_prepare.R — Read train/OOT slices from Postgres (same source as Python).
#
# Produces an in-memory list with train, validation, oot frames plus
# the target vector. Feature column list is identical to what the Python
# track uses, so the two tracks are truly comparable at reconciliation time.

suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(yaml)
  library(dplyr)
})

# Locate repo root (two levels up from this script when run via Rscript)
get_repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/")
    return(dirname(dirname(dirname(script))))
  }
  return(normalizePath("..", winslash = "/"))
}

REPO_ROOT <- get_repo_root()
cfg <- yaml::read_yaml(file.path(REPO_ROOT, "config.yml"))

connect_db <- function(cfg) {
  db <- cfg$database
  pw <- Sys.getenv("DB_PASSWORD", unset = "")
  if (nchar(pw) == 0) {
    stop("DB_PASSWORD is not set. Copy .env.example to .env and fill it in, ",
         "or export DB_PASSWORD in your shell before running.")
  }
  DBI::dbConnect(
    RPostgres::Postgres(),
    host     = Sys.getenv("DB_HOST", db$host),
    port     = as.integer(Sys.getenv("DB_PORT", db$port)),
    dbname   = Sys.getenv("DB_NAME", db$database),
    user     = Sys.getenv("DB_USER", db$user),
    password = pw
  )
}

# Features: read the feature mart (already WoE-encoded by Python pipeline).
# R track ALSO re-bins using scorecard::woebin for its own track — we pull the
# raw loan characteristics from the loans table, not the WoE-encoded mart.
read_modelling_frame <- function() {
  conn <- connect_db(cfg)
  on.exit(DBI::dbDisconnect(conn))

  sql <- "
    SELECT l.loan_id,
           l.issue_date,
           l.vintage,
           l.split,
           l.grade,
           l.sub_grade,
           l.term_months,
           l.loan_amnt,
           l.funded_amnt,
           l.int_rate,
           l.installment,
           l.annual_inc,
           l.dti,
           l.fico_range_low,
           l.fico_range_high,
           l.delinq_2yrs,
           l.inq_last_6mths,
           l.open_acc,
           l.pub_rec,
           l.revol_bal,
           l.revol_util,
           l.total_acc,
           l.home_ownership,
           l.verification_status,
           l.purpose,
           l.emp_length,
           fm.default_flag
    FROM credit_risk.loans l
    JOIN credit_risk.feature_mart fm USING (loan_id)
  "
  df <- DBI::dbGetQuery(conn, sql)
  # Make default_flag integer
  df$default_flag <- as.integer(df$default_flag)
  df$fico_mid <- (df$fico_range_low + df$fico_range_high) / 2
  # Pretty types for scorecard::woebin
  df$term_months <- as.integer(df$term_months)
  df$int_rate   <- as.numeric(df$int_rate)
  df
}

# If run directly, print shape
if (sys.nframe() == 0L) {
  df <- read_modelling_frame()
  cat(sprintf("Modelling frame: %d rows x %d cols\n", nrow(df), ncol(df)))
  cat("Split counts:\n")
  print(table(df$split))
  cat("Default rate by split:\n")
  print(aggregate(default_flag ~ split, data = df, FUN = mean))
}
