"""Thin wrapper around MLflow so the DAG doesn't talk to the tracking API directly.

Keeping this separate makes it easy to mock in tests and to swap tracking
backends (local file store in dev, a managed tracking server in prod) via
the MLFLOW_TRACKING_URI environment variable alone.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import mlflow

logger = logging.getLogger(__name__)


@contextmanager
def run(experiment_name: str, run_name: str) -> Iterator[None]:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        logger.info("Started MLflow run '%s' in experiment '%s'", run_name, experiment_name)
        yield


def log_pipeline_metrics(row_count: int, null_rate: float, duplicate_rate: float, runtime_seconds: float) -> None:
    """Log the data-quality and runtime metrics for a single pipeline run."""
    mlflow.log_metric("row_count", row_count)
    mlflow.log_metric("max_null_rate", null_rate)
    mlflow.log_metric("duplicate_rate", duplicate_rate)
    mlflow.log_metric("runtime_seconds", runtime_seconds)


def register_model_if_improved(model_uri: str, model_name: str, metric_name: str, metric_value: float, higher_is_better: bool = True) -> bool:
    """Register a new model version only if it beats the current production model.

    Returns True if the model was registered, False if the incumbent was kept.
    Prevents a retrain run from silently promoting a worse model.
    """
    client = mlflow.tracking.MlflowClient()

    try:
        current = client.get_latest_versions(model_name, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        current = []

    if not current:
        version = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(model_name, version.version, "Production")
        logger.info("No production model found; registered version %s", version.version)
        return True

    current_run = client.get_run(current[0].run_id)
    current_metric = current_run.data.metrics.get(metric_name)

    is_improved = (
        current_metric is None
        or (higher_is_better and metric_value > current_metric)
        or (not higher_is_better and metric_value < current_metric)
    )

    if is_improved:
        version = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(model_name, version.version, "Production")
        logger.info(
            "New model (%s=%.4f) beat incumbent (%s=%s); promoted version %s",
            metric_name, metric_value, metric_name, current_metric, version.version,
        )
        return True

    logger.info(
        "New model (%s=%.4f) did not beat incumbent (%s=%.4f); keeping current production model",
        metric_name, metric_value, metric_name, current_metric,
    )
    return False
