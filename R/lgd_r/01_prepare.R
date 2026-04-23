# R/lgd_r/01_prepare.R — Read defaulted-loan frame for LGD modelling.

suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(yaml)
})

get_repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  if (length(fa) > 0) {
    return(normalizePath(
      file.path(dirname(sub("^--file=", "", fa[1])), "..", ".."),
      winslash = "/"
    ))
  }
  return(normalizePath("..", winslash = "/"))
}

REPO_ROOT <- get_repo_root()
cfg <- yaml::read_yaml(file.path(REPO_ROOT, "config.yml"))

connect_db <- function(cfg) {
  db <- cfg$database
  pw <- Sys.getenv("DB_PASSWORD", unset = "")
  if (nchar(pw) == 0) {
    stop("DB_PASSWORD is not set. Copy .env.example to .env and fill it in.")
  }
  DBI::dbConnect(
    RPostgres::Postgres(),
    host = Sys.getenv("DB_HOST", db$host),
    port = as.integer(Sys.getenv("DB_PORT", db$port)),
    dbname = Sys.getenv("DB_NAME", db$database),
    user = Sys.getenv("DB_USER", db$user),
    password = pw
  )
}

read_lgd_frame <- function() {
  conn <- connect_db(cfg)
  on.exit(DBI::dbDisconnect(conn))
  sql <- "
    SELECT d.loan_id,
           d.default_date,
           d.default_type,
           d.recovery_amount,
           d.funded_amnt,
           d.lgd,
           l.grade,
           l.sub_grade,
           l.term_months,
           l.int_rate,
           l.fico_range_low,
           l.fico_range_high,
           l.annual_inc,
           l.dti,
           l.loan_amnt,
           l.purpose,
           l.home_ownership,
           l.emp_length,
           l.verification_status,
           l.vintage,
           l.split
    FROM credit_risk.defaults d
    JOIN credit_risk.loans l USING (loan_id)
  "
  df <- DBI::dbGetQuery(conn, sql)
  df$fico_mid <- (df$fico_range_low + df$fico_range_high) / 2
  df$annual_inc_log <- log1p(pmax(0, df$annual_inc))
  df
}

if (sys.nframe() == 0L) {
  df <- read_lgd_frame()
  cat(sprintf("LGD frame: %d rows x %d cols\n", nrow(df), ncol(df)))
  cat("LGD distribution:\n")
  print(summary(df$lgd))
  cat("Split counts:\n")
  print(table(df$split))
}
