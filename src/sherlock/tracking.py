"""
tracking.py
-----------
Configuration MLflow partagée (entraînement, baselines, ré-évaluation).
"""

import subprocess

from sherlock.config import cfg


def setup_mlflow() -> None:
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
