"""Postgres engine + bulk-load helpers.

Uses SQLAlchemy for metadata / DDL; for bulk INSERT we bypass the ORM and
use psycopg2's ``COPY FROM STDIN`` which is ~50x faster than row-by-row.
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from typing import Iterable, Iterator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pipeline.config import CFG, db_url
from pipeline.logging_utils import get_logger

LOG = get_logger(__name__)
_engine: Engine | None = None


def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine scoped to the configured DB."""
    global _engine
    if _engine is None:
        _engine = create_engine(db_url(), pool_pre_ping=True, future=True)
    return _engine


@contextmanager
def raw_cursor() -> Iterator:
    """Yield a psycopg2 cursor on a short-lived connection."""
    conn = get_engine().raw_connection()
    try:
        cur = conn.cursor()
        yield cur, conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def wait_for_db(max_attempts: int = 30, delay_s: float = 2.0) -> None:
    """Poll Postgres until it accepts connections (up to ~60s)."""
    import time

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            with get_engine().begin() as cx:
                cx.execute(text("SELECT 1"))
            LOG.info("Postgres reachable on attempt %d", attempt)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            LOG.info("DB not ready (attempt %d/%d)...", attempt, max_attempts)
            time.sleep(delay_s)
    raise RuntimeError(f"Postgres never became reachable: {last_err}")


def set_schema() -> None:
    """Ensure search_path is set to the project schema."""
    schema = CFG["database"]["schema"]
    with get_engine().begin() as cx:
        cx.execute(text(f"SET search_path TO {schema}, public"))


def truncate_tables(tables: Iterable[str]) -> None:
    """Wipe the given tables (in CASCADE order)."""
    schema = CFG["database"]["schema"]
    quoted = ", ".join(f"{schema}.{t}" for t in tables)
    with get_engine().begin() as cx:
        cx.execute(text(f"TRUNCATE TABLE {quoted} CASCADE"))
        LOG.info("Truncated: %s", quoted)


def copy_dataframe(df: pd.DataFrame, table: str, columns: list[str] | None = None) -> int:
    """Bulk-insert ``df`` into ``schema.table`` via COPY FROM STDIN.

    Args:
        df: DataFrame with columns matching ``table`` (order-agnostic).
        table: Unqualified table name (schema is prepended from config).
        columns: Optional explicit column order. Defaults to ``df.columns``.

    Returns:
        Number of rows loaded.
    """
    schema = CFG["database"]["schema"]
    cols = list(columns or df.columns)
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    col_sql = ", ".join(f'"{c}"' for c in cols)
    sql = (
        f"COPY {schema}.{table} ({col_sql}) "
        "FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
    )
    with raw_cursor() as (cur, _conn):
        cur.copy_expert(sql, buf)
    LOG.info("COPY %s: %d rows", table, len(df))
    return len(df)


def read_sql(sql: str, **params) -> pd.DataFrame:
    """Convenience read (handles SET search_path)."""
    schema = CFG["database"]["schema"]
    with get_engine().begin() as cx:
        cx.execute(text(f"SET search_path TO {schema}, public"))
        return pd.read_sql(text(sql), cx, params=params)
