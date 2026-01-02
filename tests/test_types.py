"""Unit tests for shared types and validation."""
# pylint: disable=import-error

from __future__ import annotations

import pandas as pd
import pytest

from sensors_anomalies.types import validate_long_df


def test_validate_long_df_ok() -> None:
    """validate_long_df should accept a minimal valid dataframe."""
    df = pd.DataFrame(
        {
            "series_id": ["s1"],
            "timestamp": [pd.Timestamp("2024-01-01")],
            "signal": ["a"],
            "value": [1.0],
        }
    )
    validate_long_df(df)


def test_validate_long_df_missing_columns() -> None:
    """validate_long_df should raise on missing columns."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError):
        validate_long_df(df)
