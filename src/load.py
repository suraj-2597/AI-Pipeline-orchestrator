"""Load transformed data into the BigQuery warehouse layer.

Writes are partitioned by `event_date` and use WRITE_APPEND with a
day-level overwrite (`overwrite_partition`) so re-runs are idempotent
instead of duplicating a day's data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadConfig:
    project: str
    dataset: str
    table: str
    partition_field: str = "event_date"
    temp_gcs_bucket: str = ""

    @property
    def full_table_id(self) -> str:
        return f"{self.project}.{self.dataset}.{self.table}"


def overwrite_partition(df: DataFrame, config: LoadConfig, partition_value: str) -> None:
    """Overwrite a single day's partition in BigQuery.

    Uses the spark-bigquery-connector under the hood. Filtering to a single
    partition value and writing with `overwrite` semantics on that slice
    keeps re-runs of a single Airflow task idempotent.
    """
    logger.info(
        "Writing partition %s=%s to %s (%d rows)",
        config.partition_field,
        partition_value,
        config.full_table_id,
        df.count(),
    )

    (
        df.write.format("bigquery")
        .option("table", config.full_table_id)
        .option("temporaryGcsBucket", config.temp_gcs_bucket)
        .option("partitionField", config.partition_field)
        .option("datePartition", partition_value.replace("-", ""))
        .mode("overwrite")
        .save()
    )

    logger.info("Load complete for partition %s", partition_value)
