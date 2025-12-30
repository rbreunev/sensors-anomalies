"""
CSV utilities for validation and normalization.

Provides helpers for reading uploaded CSV files, inferring column types,
and ensuring they conform to the canonical long format schema.
"""

from __future__ import annotations

from io import BytesIO, StringIO
from typing import Any

import pandas as pd

from sensors_anomalies.types import validate_long_df


def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Read a CSV file from uploaded bytes.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes from file upload.

    Returns
    -------
    pd.DataFrame
        Parsed dataframe.

    Raises
    ------
    ValueError
        If the file cannot be parsed as CSV, is empty, or contains no data.
    """
    if not file_bytes:
        raise ValueError("File bytes are empty or null")

    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}") from e

    if df is None:
        raise ValueError("CSV parsing returned null DataFrame")

    if df.empty:
        raise ValueError("CSV file is empty or contains no data")

    return df


def infer_timestamp_column(df: pd.DataFrame) -> str:
    """
    Infer the timestamp column name from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    str
        Name of the likely timestamp column.

    Raises
    ------
    ValueError
        If no timestamp column can be identified.
    """
    candidates = ["timestamp", "Timestamp", "time", "Time", "datetime", "Datetime", "date", "Date"]

    for col in candidates:
        if col in df.columns:
            return col

    # Check for datetime dtype columns
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        return datetime_cols[0]

    raise ValueError(
        f"Could not find timestamp column. Available columns: {df.columns.tolist()}\n"
        "Expected one of: timestamp, Timestamp, time, datetime, date, or a datetime-typed column"
    )


def infer_label_column(df: pd.DataFrame) -> str | None:
    """
    Infer the label/target column name from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    str or None
        Name of the likely label column, or None if not found.
    """
    candidates = ["label", "Label", "target", "Target", "fault", "Fault", "anomaly", "Anomaly"]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def is_long_format(df: pd.DataFrame) -> bool:
    """
    Check if a dataframe is already in long format.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    bool
        True if the dataframe has the required long-format columns.
    """
    required = {"series_id", "timestamp", "signal", "value"}
    return required.issubset(df.columns)


def validate_and_normalize_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a dataframe that is already in long format.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe expected to be in long format.

    Returns
    -------
    pd.DataFrame
        Validated and normalized dataframe.

    Raises
    ------
    ValueError
        If the dataframe does not conform to long format requirements.
    """
    # Check for required columns
    if not is_long_format(df):
        raise ValueError(
            "DataFrame is not in long format. "
            "Required columns: series_id, timestamp, signal, value. "
            f"Found: {df.columns.tolist()}"
        )

    # Parse timestamp if needed
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Validate using types module
    validate_long_df(df)

    return df


def transform_wide_to_long(
    df_wide: pd.DataFrame,
    timestamp_col: str | None = None,
    label_col: str | None = None,
    series_id: str = "default_series",
) -> pd.DataFrame:
    """
    Transform a wide-format dataframe to long format.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format dataframe with timestamp and multiple sensor columns.
    timestamp_col : str, optional
        Name of the timestamp column. If None, will be inferred.
    label_col : str, optional
        Name of the label column. If None, will be inferred.
    series_id : str, default="default_series"
        Series identifier to assign to all rows.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns: series_id, timestamp, signal, value, [label].

    Raises
    ------
    ValueError
        If timestamp column cannot be found or dataframe is invalid.
    """
    df = df_wide.copy()

    # Infer columns if not provided
    if timestamp_col is None:
        timestamp_col = infer_timestamp_column(df)

    if label_col is None:
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

    # Rename timestamp column
    df_long = df_long.rename(columns={timestamp_col: "timestamp"})
    if label_col:
        df_long = df_long.rename(columns={label_col: "label"})

    # Add series_id
    df_long["series_id"] = series_id

    # Reorder columns
    cols = ["series_id", "timestamp", "signal", "value"]
    if label_col:
        cols.append("label")
    df_long = df_long[cols]

    # Sort by timestamp and signal
    df_long = df_long.sort_values(["timestamp", "signal"]).reset_index(drop=True)

    # Validate
    validate_long_df(df_long)

    return df_long
