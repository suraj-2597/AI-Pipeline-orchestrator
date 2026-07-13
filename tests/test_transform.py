import pytest
from pyspark.sql import Row, SparkSession

from src.transform import (
    DataQualityError,
    add_event_date,
    drop_duplicates,
    enforce_quality_gate,
    fill_missing_numeric,
    profile,
)


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder.master("local[2]")
        .appName("test-transform")
        .getOrCreate()
    )


def test_drop_duplicates_removes_repeated_ids(spark):
    df = spark.createDataFrame(
        [Row(event_id="a", amount=1.0), Row(event_id="a", amount=1.0), Row(event_id="b", amount=2.0)]
    )
    result = drop_duplicates(df, subset=["event_id"])
    assert result.count() == 2


def test_fill_missing_numeric_replaces_nulls(spark):
    df = spark.createDataFrame([Row(event_id="a", amount=None), Row(event_id="b", amount=5.0)])
    result = fill_missing_numeric(df, columns=["amount"], value=0.0)
    values = sorted(r["amount"] for r in result.collect())
    assert values == [0.0, 5.0]


def test_add_event_date_derives_date_from_timestamp(spark):
    df = spark.createDataFrame([Row(event_ts="2026-06-01T12:00:00")])
    result = add_event_date(df, timestamp_col="event_ts")
    assert "event_date" in result.columns
    assert str(result.collect()[0]["event_date"]) == "2026-06-01"


def test_profile_reports_null_rate(spark):
    df = spark.createDataFrame(
        [Row(a=1, b=None), Row(a=2, b=None), Row(a=None, b="x"), Row(a=4, b="y")]
    )
    report = profile(df)
    assert report.total_rows == 4
    assert report.null_rate_by_column["b"] == 0.5
    assert report.null_rate_by_column["a"] == 0.25


def test_enforce_quality_gate_raises_on_high_null_rate(spark):
    df = spark.createDataFrame([Row(a=None), Row(a=None), Row(a=1)])
    report = profile(df)
    with pytest.raises(DataQualityError):
        enforce_quality_gate(report, max_null_rate=0.05)


def test_enforce_quality_gate_passes_clean_data(spark):
    df = spark.createDataFrame([Row(a=1), Row(a=2), Row(a=3)])
    report = profile(df)
    enforce_quality_gate(report, max_null_rate=0.05)  # should not raise
