"""
Dataset registry and transformers.

The project uses a registry to transform wide-format CSVs to the canonical
long format at runtime.
"""

from __future__ import annotations

from sensors_anomalies.datasets.registry import (
    apply_transformer,
    get_transformer,
    list_transformers,
    register_transformer,
)

__all__ = ["register_transformer", "list_transformers", "get_transformer", "apply_transformer"]
