"""
Shared types and validation helpers.

This module defines a minimal "long" time-series format that all dataset loaders
must normalize to:

- series_id: identifier of the time series (e.g. sensor group / unit / run)
- timestamp: datetime (timezone-aware recommended)
- signal: sensor name / channel
- value: numeric value
- label (optional): 0/1 for normal/fault if available
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """
    Dataset metadata.

    Parameters
    ----------
    dataset_id : str
        Unique dataset identifier used by the registry.
    name : str
        Human-readable dataset name.
    url : str
        Public URL reference.
    """

    dataset_id: str
    name: str
    url: str


@dataclass(frozen=True)
class LoadedDataset:
    """
    Loaded dataset in the normalized long format.

    Parameters
    ----------
    spec : DatasetSpec
        Dataset metadata.
    df : pandas.DataFrame
        Long-format dataframe.
    has_labels : bool
        Whether the dataset includes usable labels.
    """

    spec: DatasetSpec
    df: pd.DataFrame
    has_labels: bool


def validate_long_df(df: pd.DataFrame) -> None:
    """
    Validate the normalized long-format dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe expected to contain columns: series_id, timestamp, signal, value.

    Raises
    ------
    ValueError
        If required columns are missing or dataframe is empty.
    TypeError
        If the timestamp column is not a datetime dtype.
    """
    required = {"series_id", "timestamp", "signal", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Dataset is empty")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("timestamp must be datetime dtype")
