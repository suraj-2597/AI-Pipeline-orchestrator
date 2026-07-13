"""Extraction utilities for pulling raw training data into the lake.

Sources are read from GCS as newline-delimited JSON or Parquet and returned
as a Spark DataFrame. Kept intentionally thin: extract should not transform,
only fetch and do light schema validation so failures are attributable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractConfig:
    source_uri: str          # e.g. gs://bucket/raw/events/*.json
    file_format: str = "json"  # "json" | "parquet" | "csv"
    expected_schema: StructType | None = None


class SchemaValidationError(RuntimeError):
    """Raised when extracted data does not match the expected schema."""


def extract(spark: SparkSession, config: ExtractConfig) -> DataFrame:
    """Read raw data from `config.source_uri` into a DataFrame.

    Raises SchemaValidationError if an expected_schema was provided and the
    extracted columns don't match, so bad upstream data fails fast instead of
    silently propagating into the warehouse.
    """
    logger.info("Extracting from %s (%s)", config.source_uri, config.file_format)

    reader = spark.read
    if config.expected_schema is not None:
        reader = reader.schema(config.expected_schema)

    if config.file_format == "json":
        df = reader.json(config.source_uri)
    elif config.file_format == "parquet":
        df = reader.parquet(config.source_uri)
    elif config.file_format == "csv":
        df = reader.option("header", "true").csv(config.source_uri)
    else:
        raise ValueError(f"Unsupported file_format: {config.file_format}")

    if config.expected_schema is not None:
        actual_fields = set(df.schema.fieldNames())
        expected_fields = set(config.expected_schema.fieldNames())
        missing = expected_fields - actual_fields
        if missing:
            raise SchemaValidationError(
                f"Extracted data from {config.source_uri} is missing columns: {sorted(missing)}"
            )

    row_count = df.count()
    logger.info("Extracted %d rows from %s", row_count, config.source_uri)
    return df
