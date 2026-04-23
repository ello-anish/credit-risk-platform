-- credit-risk-platform schema
-- Executed once by postgres container on first start.

CREATE SCHEMA IF NOT EXISTS credit_risk;
SET search_path TO credit_risk, public;

-- ================================================================
-- loans: static origination data, one row per loan
-- ================================================================
CREATE TABLE IF NOT EXISTS loans (
    loan_id             BIGINT        PRIMARY KEY,
    issue_date          DATE          NOT NULL,
    term_months         INTEGER       NOT NULL,
    grade               VARCHAR(2)    NOT NULL,
    sub_grade           VARCHAR(3)    NOT NULL,
    fico_range_low      INTEGER,
    fico_range_high     INTEGER,
    annual_inc          NUMERIC(14,2),
    dti                 NUMERIC(7,2),
    purpose             VARCHAR(50),
    home_ownership      VARCHAR(20),
    emp_length          VARCHAR(20),
    verification_status VARCHAR(30),
    loan_amnt           NUMERIC(12,2),
    funded_amnt         NUMERIC(12,2),
    int_rate            NUMERIC(7,4),
    installment         NUMERIC(12,2),
    addr_state          VARCHAR(2),
    delinq_2yrs         INTEGER,
    earliest_cr_line    DATE,
    inq_last_6mths      INTEGER,
    open_acc            INTEGER,
    pub_rec             INTEGER,
    revol_bal           NUMERIC(14,2),
    revol_util          NUMERIC(7,4),
    total_acc           INTEGER,
    vintage             VARCHAR(10),              -- e.g. "2014-Q2"
    split               VARCHAR(20),              -- train / validation / oot / excluded
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_loans_issue_date ON loans(issue_date);
CREATE INDEX IF NOT EXISTS idx_loans_grade      ON loans(grade);
CREATE INDEX IF NOT EXISTS idx_loans_vintage    ON loans(vintage);
CREATE INDEX IF NOT EXISTS idx_loans_split      ON loans(split);

-- ================================================================
-- loan_status: monthly snapshots
-- Synthesized for LendingClub by linear interpolation between
-- issue_date and final status observation date. See pipeline/ingest.py.
-- ================================================================
CREATE TABLE IF NOT EXISTS loan_status (
    loan_id             BIGINT        NOT NULL,
    as_of_date          DATE          NOT NULL,
    current_status      VARCHAR(60)   NOT NULL,
    days_past_due       INTEGER       DEFAULT 0,
    outstanding_balance NUMERIC(14,2),
    PRIMARY KEY (loan_id, as_of_date),
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_status_as_of  ON loan_status(as_of_date);
CREATE INDEX IF NOT EXISTS idx_status_status ON loan_status(current_status);
CREATE INDEX IF NOT EXISTS idx_status_dpd    ON loan_status(days_past_due);

-- ================================================================
-- defaults: one row per defaulted loan
-- ================================================================
CREATE TABLE IF NOT EXISTS defaults (
    loan_id         BIGINT        PRIMARY KEY,
    default_date    DATE          NOT NULL,
    default_type    VARCHAR(50),                  -- Charged Off / Default / Late 31-120
    recovery_amount NUMERIC(14,2) DEFAULT 0,
    recovery_date   DATE,
    funded_amnt     NUMERIC(12,2),                -- cached for fast LGD computation
    lgd             NUMERIC(8,6),                 -- 1 - recovery/funded, clipped [0,1]
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_defaults_date ON defaults(default_date);

-- ================================================================
-- macro: quarterly snapshots of macroeconomic factors
-- ================================================================
CREATE TABLE IF NOT EXISTS macro (
    as_of_date    DATE          PRIMARY KEY,
    gdp_growth    NUMERIC(8,4),                   -- YoY real GDP growth (%)
    unemployment  NUMERIC(6,3),                   -- unemployment rate (%)
    hpi           NUMERIC(12,4),                  -- house price index level
    treasury_10y  NUMERIC(6,4),                   -- 10Y UST yield (%)
    vix           NUMERIC(7,4)                    -- VIX close
);

-- ================================================================
-- ecl_results: engine output, keyed by (loan, as-of, scenario)
-- ================================================================
CREATE TABLE IF NOT EXISTS ecl_results (
    loan_id       BIGINT        NOT NULL,
    as_of_date    DATE          NOT NULL,
    scenario      VARCHAR(40)   NOT NULL DEFAULT 'baseline',
    pd_12m        NUMERIC(10,8),
    pd_lifetime   NUMERIC(10,8),
    lgd           NUMERIC(10,8),
    ead           NUMERIC(14,2),
    ecl_12m       NUMERIC(14,4),
    ecl_lifetime  NUMERIC(14,4),
    reported_ecl  NUMERIC(14,4),
    stage         SMALLINT,                        -- 1, 2, or 3
    model_version VARCHAR(32),
    computed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (loan_id, as_of_date, scenario),
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ecl_stage    ON ecl_results(stage);
CREATE INDEX IF NOT EXISTS idx_ecl_scenario ON ecl_results(scenario);
CREATE INDEX IF NOT EXISTS idx_ecl_as_of    ON ecl_results(as_of_date);

-- ================================================================
-- data_quality_runs: audit log of pandera gate executions
-- ================================================================
CREATE TABLE IF NOT EXISTS data_quality_runs (
    run_id       SERIAL        PRIMARY KEY,
    run_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    table_name   VARCHAR(40)   NOT NULL,
    check_name   VARCHAR(80)   NOT NULL,
    passed       BOOLEAN       NOT NULL,
    failed_rows  INTEGER       DEFAULT 0,
    message      TEXT
);

CREATE INDEX IF NOT EXISTS idx_dq_run_at ON data_quality_runs(run_at DESC);
