"""
Algorithms registry.

Each algorithm accepts a long-format dataframe and returns a standardized result.
"""

from __future__ import annotations

from sensors_anomalies.algorithms.registry import list_algorithms, register_algorithm, run_algorithm

__all__ = ["register_algorithm", "list_algorithms", "run_algorithm"]
