"""Cleaning, feature engineering, and data-quality gating.

Everything here is a pure function of (DataFrame -> DataFrame), which keeps
the transforms unit-testable with a local SparkSession and composable inside
the Airflow DAG without hidden state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualityReport:
    total_rows: int
    null_rate_by_column: dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0

    def has_null_rate_above(self, threshold: float) -> list[str]:
        return [col for col, rate in self.null_rate_by_column.items() if rate > threshold]


class DataQualityError(RuntimeError):
    """Raised when transformed data fails a hard quality gate."""


def drop_duplicates(df: DataFrame, subset: list[str]) -> DataFrame:
    return df.dropDuplicates(subset=subset)


def fill_missing_numeric(df: DataFrame, columns: list[str], value: float = 0.0) -> DataFrame:
    return df.fillna(value, subset=columns)


def add_event_date(df: DataFrame, timestamp_col: str = "event_ts") -> DataFrame:
    """Derive a partition-friendly `event_date` column from a timestamp."""
    return df.withColumn("event_date", F.to_date(F.col(timestamp_col)))


def profile(df: DataFrame) -> QualityReport:
    """Compute null rates per column and a duplicate-row count.

    Used both as an observability step (logged every run) and as a gate
    (see `enforce_quality_gate`) before data is allowed to load.
    """
    total = df.count()
    if total == 0:
        return QualityReport(total_rows=0)

    null_counts = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).collect()[0].asDict()

    null_rates = {c: (null_counts[c] or 0) / total for c in df.columns}
    duplicate_rows = total - df.dropDuplicates().count()

    return QualityReport(total_rows=total, null_rate_by_column=null_rates, duplicate_rows=duplicate_rows)


def enforce_quality_gate(report: QualityReport, max_null_rate: float = 0.05, max_duplicate_rate: float = 0.01) -> None:
    """Raise DataQualityError if the batch fails hard thresholds.

    Called from the DAG right before the load task; a failure here stops the
    pipeline instead of writing bad data into the warehouse.
    """
    offending = report.has_null_rate_above(max_null_rate)
    if offending:
        raise DataQualityError(
            f"Columns exceed max null rate {max_null_rate:.1%}: {offending}"
        )

    if report.total_rows and (report.duplicate_rows / report.total_rows) > max_duplicate_rate:
        raise DataQualityError(
            f"Duplicate rate {report.duplicate_rows / report.total_rows:.2%} exceeds "
            f"threshold {max_duplicate_rate:.2%}"
        )

    logger.info(
        "Quality gate passed: %d rows, max null rate %.2f%%, duplicate rate %.2f%%",
        report.total_rows,
        max(report.null_rate_by_column.values(), default=0) * 100,
        (report.duplicate_rows / report.total_rows * 100) if report.total_rows else 0,
    )
