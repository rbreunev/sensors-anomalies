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

TransformerFn = Callable[[pd.DataFrame], pd.DataFrame]

_TRANSFORMERS: dict[str, tuple[DatasetSpec, TransformerFn]] = {}


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

    Raises
    ------
    KeyError
        If dataset_id is not registered.
    ValueError
        If transformation fails or output is invalid.
    """
    spec, transformer = get_transformer(dataset_id)
    df_long = transformer(df_wide)
    validate_long_df(df_long)
    return df_long


def _register_defaults() -> None:
    """Register default transformers."""
    # Register real dataset transformers
    try:
        from sensors_anomalies.datasets.sensor_fault import register_sensor_fault_transformer

        register_sensor_fault_transformer()
    except ImportError:
        pass  # Module not available yet
