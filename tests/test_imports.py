"""Import tests for the project skeleton."""

from __future__ import annotations


def test_imports() -> None:
    """Ensure the package can be imported."""
    import sensors_anomalies  # noqa: F401


def test_types_module() -> None:
    """Test types module imports."""
    from sensors_anomalies.types import DatasetSpec, LoadedDataset, validate_long_df  # noqa: F401


def test_datasets_registry() -> None:
    """Test datasets registry imports."""
    from sensors_anomalies.datasets import (  # noqa: F401
        apply_transformer,
        get_transformer,
        list_transformers,
        register_transformer,
    )


def test_algorithms_registry() -> None:
    """Test algorithms registry imports."""
    from sensors_anomalies.algorithms import (  # noqa: F401
        list_algorithms,
        register_algorithm,
        run_algorithm,
    )


def test_utils_module() -> None:
    """Test utils module imports."""
    from sensors_anomalies.utils import ensure_utc  # noqa: F401
