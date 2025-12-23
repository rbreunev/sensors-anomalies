"""
Dataset registry and loaders.

The project uses a registry so the Gradio app can switch datasets without
changing application code.
"""

from __future__ import annotations

from sensors_anomalies.datasets.registry import list_datasets, load_dataset, register_dataset

__all__ = ["register_dataset", "list_datasets", "load_dataset"]
