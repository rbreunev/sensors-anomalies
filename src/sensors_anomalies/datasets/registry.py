"""
Datasets registry.

A dataset is registered with a DatasetSpec and a loader function returning a
LoadedDataset in the common long format.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from sensors_anomalies.types import DatasetSpec, LoadedDataset, validate_long_df

LoaderFn = Callable[[], LoadedDataset]

_DATASETS: dict[str, tuple[DatasetSpec, LoaderFn]] = {}


def register_dataset(spec: DatasetSpec, loader: LoaderFn) -> None:
    """
    Register a dataset loader.

    Parameters
    ----------
    spec : DatasetSpec
        Dataset metadata.
    loader : LoaderFn
        Loader returning the dataset in long format.
    """
    _DATASETS[spec.dataset_id] = (spec, loader)


def list_datasets() -> list[str]:
    """
    List registered dataset identifiers.

    Returns
    -------
    list[str]
        Sorted dataset ids.
    """
    if not _DATASETS:
        _register_defaults()
    return sorted(_DATASETS.keys())


def load_dataset(dataset_id: str) -> LoadedDataset:
    """
    Load a dataset by id.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier.

    Returns
    -------
    LoadedDataset
        Loaded dataset.
    """
    if not _DATASETS:
        _register_defaults()
    spec, loader = _DATASETS[dataset_id]
    ds = loader()
    validate_long_df(ds.df)
    return ds


def _register_defaults() -> None:
    """Register placeholder datasets so the app runs before real loaders are added."""

    def _toy_loader() -> LoadedDataset:
        """
        Create a tiny toy dataset.

        Returns
        -------
        LoadedDataset
            Small dataset with labels.
        """
        df = pd.DataFrame(
            {
                "series_id": ["s1"] * 5,
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="min"),
                "signal": ["sensor_a"] * 5,
                "value": [1.0, 1.0, 1.0, 10.0, 1.0],
                "label": [0, 0, 0, 1, 0],
            }
        )
        return LoadedDataset(
            spec=DatasetSpec("toy", "Toy dataset", "N/A"),
            df=df,
            has_labels=True,
        )

    register_dataset(DatasetSpec("toy", "Toy dataset", "N/A"), _toy_loader)
