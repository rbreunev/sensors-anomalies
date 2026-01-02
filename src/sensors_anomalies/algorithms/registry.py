"""
Algorithms registry.

Algorithms are registered with a spec and a callable. The callable takes the
normalized long-format dataframe and returns a result dataframe with columns:

- series_id
- timestamp_start
- timestamp_end
- score
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AlgoSpec:
    """
    Algorithm metadata.

    Parameters
    ----------
    algo_id : str
        Unique algorithm identifier used by the registry.
    name : str
        Human-readable algorithm name.
    kind : str
        Algorithm family (rules|unsupervised|supervised).
    """

    algo_id: str
    name: str
    kind: str


AlgoFn = Callable[[pd.DataFrame], pd.DataFrame]
_ALGOS: dict[str, tuple[AlgoSpec, AlgoFn]] = {}


def register_algorithm(spec: AlgoSpec, fn: AlgoFn) -> None:
    """
    Register an algorithm.

    Parameters
    ----------
    spec : AlgoSpec
        Algorithm metadata.
    fn : AlgoFn
        Callable implementing the algorithm.
    """
    _ALGOS[spec.algo_id] = (spec, fn)


def list_algorithms() -> list[str]:
    """
    List registered algorithm identifiers.

    Returns
    -------
    list[str]
        Sorted algorithm ids.
    """
    if not _ALGOS:
        _register_defaults()
    return sorted(_ALGOS.keys())


def run_algorithm(algo_id: str, df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Run a registered algorithm on the provided data.

    Parameters
    ----------
    algo_id : str
        Registered algorithm id.
    df_long : pd.DataFrame
        Normalized long-format dataframe.

    Returns
    -------
    pd.DataFrame
        Result dataframe.
    """
    if not _ALGOS:
        _register_defaults()
    _, fn = _ALGOS[algo_id]
    return fn(df_long)


def _register_defaults() -> None:
    """Register default anomaly detection algorithms."""
    # pylint: disable=import-outside-toplevel
    from sensors_anomalies.algorithms.unsupervised import (
        detect_iqr,
        detect_isolation_forest,
        detect_zscore,
    )

    register_algorithm(
        AlgoSpec("zscore", "Z-Score", "unsupervised"),
        detect_zscore,
    )

    register_algorithm(
        AlgoSpec("iqr", "IQR (Interquartile Range)", "unsupervised"),
        detect_iqr,
    )

    register_algorithm(
        AlgoSpec("isolation_forest", "Isolation Forest", "unsupervised"),
        detect_isolation_forest,
    )
