"""
CSV utilities for validation and normalization.

Provides helpers for reading uploaded CSV files, inferring column types,
and ensuring they conform to the canonical long format schema.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from sensors_anomalies.types import validate_long_df


def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Read a CSV file from uploaded bytes.

    Automatically detects CSV separator (comma ',' or semicolon ';').
    Decimal separator must be a period '.'.

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

    # Try to detect separator by reading first line
    try:
        first_line = file_bytes.split(b"\n")[0].decode("utf-8")
        comma_count = first_line.count(",")
        semicolon_count = first_line.count(";")

        # Choose separator with more occurrences
        separator = ";" if semicolon_count > comma_count else ","
    except Exception:  # pylint: disable=broad-exception-caught
        # Default to comma if detection fails
        separator = ","

    # Try to parse CSV with detected separator
    try:
        df = pd.read_csv(BytesIO(file_bytes), sep=separator, decimal=".")
    except Exception as e:
        raise ValueError(
            f"Failed to parse CSV: {e}\n"
            "Supported formats:\n"
            "  • Separator: comma ',' or semicolon ';' (auto-detected)\n"
            "  • Decimal separator: period '.' only"
        ) from e

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
    candidates = [
        "timestamp",
        "Timestamp",
        "time",
        "Time",
        "datetime",
        "Datetime",
        "date",
        "Date",
        "Horodatage",
        "capture_date",
        "measurement_date",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Check for datetime dtype columns
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        return datetime_cols[0]

    raise ValueError(
        f"Could not find timestamp column. Available columns: {df.columns.tolist()}\n"
        "Acceptable timestamp column names: timestamp, Timestamp, time, Time, datetime, Datetime, date, Date, "
        "Horodatage, capture_date, measurement_date"
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
    str | None
        Name of the likely label column, or None if not found.
    """
    candidates = ["label", "Label", "target", "Target", "fault", "Fault", "anomaly", "Anomaly"]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def infer_signal_column(df: pd.DataFrame) -> str | None:
    """
    Infer the signal/sensor ID column name from a dataframe.

    This is used to detect semi-long format CSVs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    str | None
        Name of the likely signal column, or None if not found.
    """
    candidates = [
        "signal",
        "Signal",
        "sensor",
        "Sensor",
        "sensor_id",
        "SensorId",
        "SensorID",
        "sensor_name",
        "SensorName",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def infer_value_column(df: pd.DataFrame) -> str | None:
    """
    Infer the value/measurement column name from a dataframe.

    This is used to detect semi-long format CSVs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    str | None
        Name of the likely value column, or None if not found.
    """
    candidates = ["value", "Value", "measurement", "Measurement", "reading", "Reading"]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def is_semi_long_format(df: pd.DataFrame) -> bool:
    """
    Check if a dataframe is in semi-long format.

    Semi-long format has timestamp, signal/sensor ID, and value columns
    but is missing the series_id column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    bool
        True if the dataframe appears to be in semi-long format.
    """
    try:
        infer_timestamp_column(df)  # Check if timestamp column exists
    except ValueError:
        return False

    signal_col = infer_signal_column(df)
    value_col = infer_value_column(df)

    return signal_col is not None and value_col is not None


def normalize_semi_long_format(df: pd.DataFrame, series_id: str = "default_series") -> pd.DataFrame:
    """
    Normalize a semi-long format dataframe to full long format.

    Semi-long format has timestamp, signal, and value columns but no series_id.
    This function adds the series_id column and renames columns to match schema.

    Parameters
    ----------
    df : pd.DataFrame
        Semi-long format dataframe.
    series_id : str, optional
        Series identifier to assign to all rows, by default "default_series".

    Returns
    -------
    pd.DataFrame
        Full long-format dataframe with series_id added.

    Raises
    ------
    ValueError
        If required columns cannot be found.
    """
    df_out = df.copy()

    # Infer column names
    timestamp_col = infer_timestamp_column(df_out)
    signal_col = infer_signal_column(df_out)
    value_col = infer_value_column(df_out)
    label_col = infer_label_column(df_out)

    if signal_col is None or value_col is None:
        raise ValueError(
            f"Cannot normalize semi-long format. Found columns: {df_out.columns.tolist()}\n"
            "Expected signal column (sensor, SensorId, etc.) and value column (value, Value, etc.)"
        )

    # Parse timestamp
    df_out[timestamp_col] = pd.to_datetime(df_out[timestamp_col])

    # Rename columns to standard names
    rename_map = {
        timestamp_col: "timestamp",
        signal_col: "signal",
        value_col: "value",
    }
    if label_col:
        rename_map[label_col] = "label"

    df_out = df_out.rename(columns=rename_map)

    # Add series_id
    df_out["series_id"] = series_id

    # Reorder columns
    cols = ["series_id", "timestamp", "signal", "value"]
    if label_col:
        cols.append("label")

    # Keep only relevant columns
    df_out = df_out[[col for col in cols if col in df_out.columns]]

    # Sort by timestamp and signal
    df_out = df_out.sort_values(["timestamp", "signal"]).reset_index(drop=True)

    # Validate
    validate_long_df(df_out)

    return df_out


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

    Automatically detects if the data is in semi-long format (has signal and value columns)
    or true wide format (has multiple sensor columns).

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format or semi-long format dataframe.
    timestamp_col : str | None, optional
        Name of the timestamp column. If None, will be inferred, by default None.
    label_col : str | None, optional
        Name of the label column. If None, will be inferred, by default None.
    series_id : str, optional
        Series identifier to assign to all rows, by default "default_series".

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns: series_id, timestamp, signal, value, [label].

    Raises
    ------
    ValueError
        If timestamp column cannot be found or dataframe is invalid.
    """
    # Check if already in semi-long format
    if is_semi_long_format(df_wide):
        return normalize_semi_long_format(df_wide, series_id=series_id)

    # Otherwise, proceed with wide-to-long transformation
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

    sensor_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    if not sensor_cols:
        raise ValueError(f"No numeric sensor columns found. Excluded: {exclude_cols}, Available: {df.columns.tolist()}")

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
