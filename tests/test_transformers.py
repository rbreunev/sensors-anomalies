"""
Tests for dataset transformers.

Validates that registered transformers correctly convert wide-format data
to the canonical long format.
"""
# ruff: noqa: S101
# pylint: disable=import-error

from __future__ import annotations

import pandas as pd
import pytest

from sensors_anomalies.datasets.registry import (
    apply_transformer,
    get_transformer,
    list_transformers,
    transform_sensor_fault,
)
from sensors_anomalies.types import validate_long_df


def test_list_transformers() -> None:
    """Test that transformer listing returns expected transformers."""
    transformers = list_transformers()
    assert isinstance(transformers, list)
    assert "sensor_fault" in transformers


def test_get_transformer() -> None:
    """Test retrieving a registered transformer."""
    spec, transformer_fn = get_transformer("sensor_fault")

    assert spec.dataset_id == "sensor_fault"
    assert spec.name == "Sensor Fault Detection (Kaggle)"
    assert callable(transformer_fn)


def test_get_transformer_invalid() -> None:
    """Test that requesting invalid transformer raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        get_transformer("nonexistent_transformer")


def test_sensor_fault_transformer() -> None:
    """Test sensor fault transformer with mock data."""
    # Create mock wide-format data similar to Kaggle dataset
    df_wide = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "sensor_00": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sensor_01": [1.1, 2.1, 3.1, 4.1, 5.1],
            "sensor_02": [1.2, 2.2, 3.2, 4.2, 5.2],
            "target": [0, 0, 0, 1, 0],
        }
    )

    df_long = transform_sensor_fault(df_wide)

    # Validate schema
    validate_long_df(df_long)

    # Check structure
    assert "series_id" in df_long.columns
    assert "timestamp" in df_long.columns
    assert "signal" in df_long.columns
    assert "value" in df_long.columns
    assert "label" in df_long.columns

    # Check dimensions
    assert len(df_long) == 5 * 3  # 5 timestamps * 3 sensors
    assert df_long["series_id"].unique()[0] == "sensor_fault_series"
    assert set(df_long["signal"].unique()) == {"sensor_00", "sensor_01", "sensor_02"}

    # Check label preservation
    assert "label" in df_long.columns
    timestamp_with_fault = df_wide["timestamp"].iloc[3]
    fault_rows = df_long[df_long["timestamp"] == timestamp_with_fault]
    assert all(fault_rows["label"] == 1)


def test_sensor_fault_transformer_no_labels() -> None:
    """Test sensor fault transformer without label column."""
    df_wide = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "sensor_00": [1.0, 2.0, 3.0],
            "sensor_01": [1.1, 2.1, 3.1],
        }
    )

    df_long = transform_sensor_fault(df_wide)

    validate_long_df(df_long)
    assert "label" not in df_long.columns
    assert len(df_long) == 3 * 2


def test_sensor_fault_transformer_various_timestamp_names() -> None:
    """Test that transformer handles various timestamp column names."""
    for timestamp_col in ["timestamp", "Timestamp", "time", "datetime"]:
        df_wide = pd.DataFrame(
            {
                timestamp_col: pd.date_range("2024-01-01", periods=2),
                "sensor_00": [1.0, 2.0],
            }
        )

        df_long = transform_sensor_fault(df_wide)
        validate_long_df(df_long)
        assert "timestamp" in df_long.columns


def test_apply_transformer() -> None:
    """Test applying a transformer through the registry."""
    df_wide = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "sensor_00": [1.0, 2.0, 3.0],
            "sensor_01": [1.1, 2.1, 3.1],
        }
    )

    df_long = apply_transformer("sensor_fault", df_wide)

    validate_long_df(df_long)
    assert len(df_long) == 3 * 2


def test_apply_transformer_invalid() -> None:
    """Test that applying invalid transformer raises KeyError."""
    df_wide = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2),
            "sensor_00": [1.0, 2.0],
        }
    )

    with pytest.raises(KeyError):
        apply_transformer("nonexistent", df_wide)


def test_sensor_fault_transformer_no_sensors() -> None:
    """Test that transformer fails gracefully with no sensor columns."""
    df_wide = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2),
            "text_column": ["a", "b"],
        }
    )

    with pytest.raises(ValueError, match="No numeric sensor columns found"):
        transform_sensor_fault(df_wide)


def test_sensor_fault_transformer_no_timestamp() -> None:
    """Test that transformer fails when timestamp column is missing."""
    df_wide = pd.DataFrame(
        {
            "sensor_00": [1.0, 2.0, 3.0],
            "sensor_01": [1.1, 2.1, 3.1],
        }
    )

    with pytest.raises(ValueError, match="Could not find timestamp column"):
        transform_sensor_fault(df_wide)
