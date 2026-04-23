"""Config loader — single source of truth for paths, DB creds, and hyperparams.

Loads config.yml at import time from the repo root (discovered by walking
upward until a config.yml is found). Public access pattern:

    from pipeline.config import CFG, REPO_ROOT
    CFG["database"]["port"]
    REPO_ROOT / "data" / "raw"
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def _find_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` until a ``config.yml`` is found."""
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "config.yml").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate config.yml walking up from {start}. "
        "Ensure you run from within the credit-risk-platform repo."
    )


REPO_ROOT: Path = _find_repo_root(Path(__file__).parent)


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Read and cache ``config.yml``."""
    with (REPO_ROOT / "config.yml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG: Dict[str, Any] = load_config()


def db_url() -> str:
    """Return the SQLAlchemy Postgres URL, honoring env-var overrides.

    Env vars ``DB_HOST``, ``DB_PORT``, ``DB_USER``, ``DB_PASSWORD``, ``DB_NAME``
    take precedence over config.yml (for CI / container overrides).
    """
    db = CFG["database"]
    host = os.environ.get("DB_HOST", db["host"])
    port = os.environ.get("DB_PORT", db["port"])
    user = os.environ.get("DB_USER", db["user"])
    pw = os.environ.get("DB_PASSWORD", db["password"])
    name = os.environ.get("DB_NAME", db["database"])
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{name}"


def artifacts_dir(subdir: str | None = None) -> Path:
    """Return a resolved artifacts directory, creating it if needed."""
    base = REPO_ROOT / CFG["paths"]["artifacts_dir"]
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base


def raw_path(key: str) -> Path:
    """Resolve a path from the ``paths`` section of config.yml."""
    return REPO_ROOT / CFG["paths"][key]
