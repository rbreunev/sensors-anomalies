"""
Sensor Fault Detection dataset transformer.

Reference transformer for the Kaggle Sensor Fault Detection dataset:
https://www.kaggle.com/datasets/arashnic/sensor-fault-detection-data

This module demonstrates how to transform this specific wide-format CSV
to the canonical long format. Users must download the CSV from Kaggle
and upload it to the Gradio app at runtime.

Expected input structure (wide format):
    - timestamp column (various names: timestamp, Timestamp, etc.)
    - multiple sensor columns (e.g., sensor_00, sensor_01, ..., sensor_50)
    - optional target/fault column (binary 0/1 labels)

Output structure (long format):
    - series_id: "sensor_fault_series"
    - timestamp: datetime
    - signal: sensor column name
    - value: sensor reading
    - label: (optional) fault indicator
"""

from __future__ import annotations

import pandas as pd

from sensors_anomalies.datasets.registry import register_transformer
from sensors_anomalies.types import DatasetSpec
from sensors_anomalies.utils.csv_utils import (
    infer_label_column,
    infer_timestamp_column,
    validate_long_df,
)


def transform_sensor_fault(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Transform Kaggle Sensor Fault Detection CSV to long format.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format dataframe from the Kaggle dataset.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns: series_id, timestamp, signal, value, [label].

    Raises
    ------
    ValueError
        If required columns are missing or transformation fails.
    """
    df = df_wide.copy()

    # Infer column names
    timestamp_col = infer_timestamp_column(df)
    label_col = infer_label_column(df)

    # Parse timestamp
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Identify sensor columns (all numeric columns except timestamp and label)
    exclude_cols = {timestamp_col}
    if label_col:
        exclude_cols.add(label_col)

    sensor_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not sensor_cols:
        raise ValueError(
            f"No numeric sensor columns found. Excluded: {exclude_cols}, "
            f"Available: {df.columns.tolist()}"
        )

    # Melt to long format
    id_vars = [timestamp_col]
    if label_col:
        id_vars.append(label_col)

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=sensor_cols,
        var_name="signal",
        value_name="value",
    )

    # Rename columns to match schema
    df_long = df_long.rename(columns={timestamp_col: "timestamp"})
    if label_col:
        df_long = df_long.rename(columns={label_col: "label"})

    # Add series_id (single series for this dataset)
    df_long["series_id"] = "sensor_fault_series"

    # Reorder columns
    cols = ["series_id", "timestamp", "signal", "value"]
    if label_col:
        cols.append("label")
    df_long = df_long[cols]

    # Sort by timestamp and signal for cleaner output
    df_long = df_long.sort_values(["timestamp", "signal"]).reset_index(drop=True)

    # Validate
    validate_long_df(df_long)

    return df_long


def register_sensor_fault_transformer() -> None:
    """Register the Sensor Fault Detection transformer."""
    register_transformer(
        spec=DatasetSpec(
            dataset_id="sensor_fault",
            name="Sensor Fault Detection (Kaggle)",
            url="https://www.kaggle.com/datasets/arashnic/sensor-fault-detection-data",
        ),
        transformer=transform_sensor_fault,
    )
