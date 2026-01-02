"""
Datasets registry.

A dataset transformer is registered with a DatasetSpec and a transformation function
that converts a specific wide-format CSV structure to the canonical long format.

These transformers serve as reference implementations showing how to normalize
different Kaggle dataset structures. Users upload CSVs at runtime rather than
including data in the repository.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from sensors_anomalies.types import DatasetSpec, validate_long_df
from sensors_anomalies.utils.csv_utils import infer_label_column, infer_timestamp_column

TransformerFn = Callable[[pd.DataFrame], pd.DataFrame]

_TRANSFORMERS: dict[str, tuple[DatasetSpec, TransformerFn]] = {}


# ============================================================================
# Transformer Functions
# ============================================================================


def transform_sensor_fault(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Transform Kaggle Sensor Fault Detection CSV to long format.

    Reference transformer for:
    https://www.kaggle.com/datasets/arashnic/sensor-fault-detection-data

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


# ============================================================================
# Registry Functions
# ============================================================================


def register_transformer(spec: DatasetSpec, transformer: TransformerFn) -> None:
    """
    Register a dataset transformer.

    Parameters
    ----------
    spec : DatasetSpec
        Dataset metadata.
    transformer : TransformerFn
        Function that transforms wide-format DataFrame to long format.
    """
    _TRANSFORMERS[spec.dataset_id] = (spec, transformer)


def list_transformers() -> list[str]:
    """
    List registered dataset transformer identifiers.

    Returns
    -------
    list[str]
        Sorted transformer ids.
    """
    if not _TRANSFORMERS:
        _register_defaults()
    return sorted(_TRANSFORMERS.keys())


def get_transformer(dataset_id: str) -> tuple[DatasetSpec, TransformerFn]:
    """
    Get a dataset transformer by id.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier.

    Returns
    -------
    tuple[DatasetSpec, TransformerFn]
        Dataset spec and transformer function.

    Raises
    ------
    KeyError
        If dataset_id is not registered.
    """
    if not _TRANSFORMERS:
        _register_defaults()

    if dataset_id not in _TRANSFORMERS:
        available = ", ".join(sorted(_TRANSFORMERS.keys()))
        raise KeyError(f"Transformer '{dataset_id}' not found. Available: {available}")

    return _TRANSFORMERS[dataset_id]


def apply_transformer(dataset_id: str, df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a registered transformer to a wide-format dataframe.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier.
    df_wide : pd.DataFrame
        Wide-format dataframe to transform.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe.
    """
    _spec, transformer = get_transformer(dataset_id)
    df_long = transformer(df_wide)
    validate_long_df(df_long)
    return df_long


def _register_defaults() -> None:
    """Register default transformers."""
    # Register Sensor Fault Detection transformer
    register_transformer(
        spec=DatasetSpec(
            dataset_id="sensor_fault",
            name="Sensor Fault Detection (Kaggle)",
            url="https://www.kaggle.com/datasets/arashnic/sensor-fault-detection-data",
        ),
        transformer=transform_sensor_fault,
    )
