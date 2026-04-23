"""Shared logging setup."""

from __future__ import annotations

import logging
import sys

from pipeline.config import CFG


def get_logger(name: str) -> logging.Logger:
    """Return a project-configured logger (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(CFG["logging"]["level"])
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(CFG["logging"]["format"]))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
