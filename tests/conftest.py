"""Shared fixtures for the pytest suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Make repo root importable when pytest runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
