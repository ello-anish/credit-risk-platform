"""MLflow helpers — keep experiment names consistent across Python and R."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from pipeline.config import CFG, REPO_ROOT


def setup_mlflow() -> None:
    """Point MLflow at the local ``./mlruns`` directory."""
    import mlflow
    uri = CFG["mlflow"]["tracking_uri"]
    if uri.startswith("./"):
        uri = str(REPO_ROOT / uri.lstrip("./"))
    mlflow.set_tracking_uri(uri)


@contextmanager
def mlflow_run(experiment_key: str, run_name: str) -> Iterator:
    """Context manager yielding an active mlflow run."""
    import mlflow
    setup_mlflow()
    exp_name = CFG["mlflow"]["experiments"][experiment_key]
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name) as run:
        yield run
