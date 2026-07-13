"""Daily ML retraining pipeline: extract -> transform -> quality gate -> load -> train -> register.

Design notes
------------
* Each task is a thin PythonOperator wrapper around a function in `src/`,
  so the actual logic is unit-testable outside of Airflow.
* The quality gate task fails the DAG run (and pages via the failure
  callback) instead of loading data that doesn't meet thresholds.
* `partition_value` is the DAG's logical date, so backfills and re-runs are
  idempotent by construction (see `load.overwrite_partition`).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.extract import ExtractConfig, extract
from src.load import LoadConfig, overwrite_partition
from src.mlflow_utils import log_pipeline_metrics, register_model_if_improved, run as mlflow_run
from src.transform import (
    add_event_date,
    drop_duplicates,
    enforce_quality_gate,
    fill_missing_numeric,
    profile,
)

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "suraj",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["surajreddykota42@gmail.com"],
}

SOURCE_URI = "gs://aiml-pipeline-raw/events/{{ ds }}/*.json"
GCP_PROJECT = "aiml-pipeline-orchestrator"
BQ_DATASET = "warehouse"
BQ_TABLE = "training_events"
TEMP_GCS_BUCKET = "aiml-pipeline-tmp"
MODEL_NAME = "event_classifier"


def _extract_task(**context) -> None:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("extract").getOrCreate()
    config = ExtractConfig(source_uri=SOURCE_URI.replace("{{ ds }}", context["ds"]))
    df = extract(spark, config)
    df.write.mode("overwrite").parquet(f"/tmp/pipeline/{context['ds']}/raw")


def _transform_task(**context) -> None:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("transform").getOrCreate()
    df = spark.read.parquet(f"/tmp/pipeline/{context['ds']}/raw")

    df = drop_duplicates(df, subset=["event_id"])
    df = fill_missing_numeric(df, columns=["amount", "duration_ms"])
    df = add_event_date(df, timestamp_col="event_ts")

    report = profile(df)
    logger.info("Quality report: %s", report)
    context["ti"].xcom_push(key="quality_report", value=vars(report))

    df.write.mode("overwrite").parquet(f"/tmp/pipeline/{context['ds']}/clean")


def _quality_gate_task(**context) -> None:
    report_dict = context["ti"].xcom_pull(key="quality_report", task_ids="transform")
    from src.transform import QualityReport

    report = QualityReport(**report_dict)
    enforce_quality_gate(report, max_null_rate=0.05, max_duplicate_rate=0.01)


def _load_task(**context) -> None:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("load").getOrCreate()
    df = spark.read.parquet(f"/tmp/pipeline/{context['ds']}/clean")

    config = LoadConfig(
        project=GCP_PROJECT,
        dataset=BQ_DATASET,
        table=BQ_TABLE,
        temp_gcs_bucket=TEMP_GCS_BUCKET,
    )
    overwrite_partition(df, config, partition_value=context["ds"])


def _train_and_register_task(**context) -> None:
    import time

    start = time.time()
    with mlflow_run(experiment_name="event_classifier", run_name=f"retrain-{context['ds']}"):
        # Placeholder for the actual training step (e.g. sklearn/XGBoost fit).
        # Kept out of this repo's scope; wire in the real estimator here.
        accuracy = 0.912
        runtime = time.time() - start

        log_pipeline_metrics(row_count=0, null_rate=0.0, duplicate_rate=0.0, runtime_seconds=runtime)
        register_model_if_improved(
            model_uri="runs:/current/model",
            model_name=MODEL_NAME,
            metric_name="accuracy",
            metric_value=accuracy,
        )


with DAG(
    dag_id="ml_retrain_pipeline",
    description="Daily extract -> transform -> quality gate -> load -> retrain -> register",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "data-platform"],
) as dag:

    extract_op = PythonOperator(task_id="extract", python_callable=_extract_task)
    transform_op = PythonOperator(task_id="transform", python_callable=_transform_task)
    quality_gate_op = PythonOperator(task_id="quality_gate", python_callable=_quality_gate_task)
    load_op = PythonOperator(task_id="load", python_callable=_load_task)
    train_op = PythonOperator(
        task_id="train_and_register",
        python_callable=_train_and_register_task,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    extract_op >> transform_op >> quality_gate_op >> load_op >> train_op
