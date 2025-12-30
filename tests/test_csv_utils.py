"""
Tests for CSV utilities.

Validates CSV reading, column inference, format detection, and transformations.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from sensors_anomalies.utils.csv_utils import (
    infer_label_column,
    infer_timestamp_column,
    is_long_format,
    read_uploaded_csv,
    transform_wide_to_long,
    validate_and_normalize_long_format,
)


def test_read_uploaded_csv() -> None:
    """Test reading CSV from bytes."""
    csv_data = b"timestamp,value\n2024-01-01,1.0\n2024-01-02,2.0"
    df = read_uploaded_csv(csv_data)

    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "value"]


def test_read_uploaded_csv_invalid() -> None:
    """Test that invalid CSV raises ValueError."""
    invalid_data = 'a,b\n"1,2\n3,4'
    # Should still parse, but let's test truly invalid data
    with pytest.raises(ValueError, match="Failed to parse CSV"):
        read_uploaded_csv(file_bytes=invalid_data)


def test_read_uploaded_csv_empty_bytes() -> None:
    """Test that empty bytes raises ValueError."""
    with pytest.raises(ValueError, match="File bytes are empty or null"):
        read_uploaded_csv(file_bytes=b"")


def test_read_uploaded_csv_empty_dataframe() -> None:
    """Test that CSV with no data raises ValueError."""
    # CSV with only headers, no data rows
    empty_csv = b"timestamp,value\n"
    with pytest.raises(ValueError, match="CSV file is empty or contains no data"):
        read_uploaded_csv(file_bytes=empty_csv)


def test_infer_timestamp_column() -> None:
    """Test timestamp column inference."""
    df = pd.DataFrame({"timestamp": [1, 2], "value": [3, 4]})
    assert infer_timestamp_column(df) == "timestamp"

    df = pd.DataFrame({"Timestamp": [1, 2], "value": [3, 4]})
    assert infer_timestamp_column(df) == "Timestamp"

    df = pd.DataFrame({"time": [1, 2], "value": [3, 4]})
    assert infer_timestamp_column(df) == "time"


def test_infer_timestamp_column_missing() -> None:
    """Test that missing timestamp column raises ValueError."""
    df = pd.DataFrame({"sensor_a": [1, 2], "sensor_b": [3, 4]})
    with pytest.raises(ValueError, match="Could not find timestamp column"):
        infer_timestamp_column(df)


def test_infer_label_column() -> None:
    """Test label column inference."""
    df = pd.DataFrame({"timestamp": [1, 2], "label": [0, 1]})
    assert infer_label_column(df) == "label"

    df = pd.DataFrame({"timestamp": [1, 2], "target": [0, 1]})
    assert infer_label_column(df) == "target"

    df = pd.DataFrame({"timestamp": [1, 2], "fault": [0, 1]})
    assert infer_label_column(df) == "fault"


def test_infer_label_column_missing() -> None:
    """Test that missing label column returns None."""
    df = pd.DataFrame({"timestamp": [1, 2], "value": [3, 4]})
    assert infer_label_column(df) is None


def test_is_long_format() -> None:
    """Test long format detection."""
    df_long = pd.DataFrame({
        "series_id": ["s1"] * 3,
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "signal": ["sensor_a"] * 3,
        "value": [1.0, 2.0, 3.0],
    })
    assert is_long_format(df_long) is True

    df_wide = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "sensor_a": [1.0, 2.0, 3.0],
        "sensor_b": [4.0, 5.0, 6.0],
    })
    assert is_long_format(df_wide) is False


def test_validate_and_normalize_long_format() -> None:
    """Test validation and normalization of long format data."""
    df = pd.DataFrame({
        "series_id": ["s1"] * 3,
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "signal": ["sensor_a"] * 3,
        "value": [1.0, 2.0, 3.0],
    })

    df_normalized = validate_and_normalize_long_format(df)

    assert pd.api.types.is_datetime64_any_dtype(df_normalized["timestamp"])
    assert len(df_normalized) == 3


def test_validate_and_normalize_long_format_invalid() -> None:
    """Test that invalid long format raises ValueError."""
    df = pd.DataFrame({
        "timestamp": ["2024-01-01", "2024-01-02"],
        "sensor_a": [1.0, 2.0],
    })

    with pytest.raises(ValueError, match="not in long format"):
        validate_and_normalize_long_format(df)


def test_transform_wide_to_long() -> None:
    """Test transformation from wide to long format."""
    df_wide = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "sensor_a": [1.0, 2.0, 3.0],
        "sensor_b": [4.0, 5.0, 6.0],
    })

    df_long = transform_wide_to_long(df_wide, series_id="test_series")

    assert is_long_format(df_long)
    assert len(df_long) == 6  # 3 timestamps * 2 sensors
    assert df_long["series_id"].unique()[0] == "test_series"
    assert set(df_long["signal"].unique()) == {"sensor_a", "sensor_b"}


def test_transform_wide_to_long_with_labels() -> None:
    """Test transformation with label column."""
    df_wide = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "sensor_a": [1.0, 2.0, 3.0],
        "sensor_b": [4.0, 5.0, 6.0],
        "label": [0, 0, 1],
    })

    df_long = transform_wide_to_long(df_wide)

    assert "label" in df_long.columns
    assert len(df_long) == 6
    # Each timestamp should have its label repeated for both sensors
    assert df_long[df_long["timestamp"] == df_wide["timestamp"].iloc[2]]["label"].unique()[0] == 1


def test_transform_wide_to_long_auto_infer() -> None:
    """Test auto-inference of timestamp and label columns."""
    df_wide = pd.DataFrame({
        "Timestamp": pd.date_range("2024-01-01", periods=2),
        "sensor_01": [1.0, 2.0],
        "sensor_02": [3.0, 4.0],
        "target": [0, 1],
    })

    df_long = transform_wide_to_long(df_wide)

    assert is_long_format(df_long)
    assert "label" in df_long.columns
    assert len(df_long) == 4


def test_transform_wide_to_long_no_sensors() -> None:
    """Test that wide format with no numeric columns raises ValueError."""
    df_wide = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2),
        "text_col": ["a", "b"],
    })

    with pytest.raises(ValueError, match="No numeric sensor columns found"):
        transform_wide_to_long(df_wide)
