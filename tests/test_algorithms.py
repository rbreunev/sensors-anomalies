"""Tests for unsupervised anomaly detection algorithms."""
# ruff: noqa: S101
# pylint: disable=import-error,redefined-outer-name,duplicate-code

from __future__ import annotations

import pandas as pd
import pytest

from sensors_anomalies.algorithms.unsupervised import (
    detect_iqr,
    detect_isolation_forest,
    detect_zscore,
)


@pytest.fixture
def simple_df_long() -> pd.DataFrame:
    """Create a simple long-format test dataframe."""
    return pd.DataFrame(
        {
            "series_id": ["series_1"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "signal": ["sensor_A"] * 10,
            "value": [1.0, 2.0, 1.5, 2.5, 1.8, 100.0, 2.2, 1.9, 2.1, 2.3],  # 100.0 is an outlier
        }
    )


@pytest.fixture
def multi_signal_df_long() -> pd.DataFrame:
    """Create a test dataframe with multiple signals."""
    n_points = 20
    return pd.DataFrame(
        {
            "series_id": ["series_1"] * n_points * 2,
            "timestamp": (pd.date_range("2024-01-01", periods=n_points, freq="h").tolist() * 2),
            "signal": ["sensor_A"] * n_points + ["sensor_B"] * n_points,
            # One outlier per signal
            "value": ([1.0] * (n_points - 1) + [10.0] + [5.0] * (n_points - 1) + [50.0]),
        }
    )


@pytest.fixture
def empty_df_long() -> pd.DataFrame:
    """Create an empty long-format dataframe."""
    return pd.DataFrame(columns=["series_id", "timestamp", "signal", "value"])


class TestZScore:
    """Tests for Z-Score anomaly detection."""

    def test_zscore_detects_outlier(self, simple_df_long: pd.DataFrame) -> None:
        """Test that Z-Score detects the obvious outlier."""
        result = detect_zscore(simple_df_long, threshold=3.0)

        assert not result.empty
        assert "is_anomaly" in result.columns
        assert "score" in result.columns

        # The outlier (100.0) should be detected
        anomalies = result[result["is_anomaly"]]
        assert len(anomalies) > 0

        # Check that the highest score corresponds to the outlier value
        max_score_idx = result["score"].idxmax()
        assert result.loc[max_score_idx, "is_anomaly"]

    def test_zscore_threshold_parameter(self, simple_df_long: pd.DataFrame) -> None:
        """Test that different thresholds affect anomaly detection."""
        result_strict = detect_zscore(simple_df_long, threshold=5.0)
        result_lenient = detect_zscore(simple_df_long, threshold=1.0)

        n_anomalies_strict = result_strict["is_anomaly"].sum()
        n_anomalies_lenient = result_lenient["is_anomaly"].sum()

        # Lenient threshold should detect more anomalies
        assert n_anomalies_lenient >= n_anomalies_strict

    def test_zscore_multi_signal(self, multi_signal_df_long: pd.DataFrame) -> None:
        """Test Z-Score on multiple signals."""
        # Use a lower threshold to detect outliers in this dataset
        result = detect_zscore(multi_signal_df_long, threshold=2.5)

        # Should have results for both signals
        signals = result["signal"].unique()
        assert len(signals) == 2
        assert "sensor_A" in signals
        assert "sensor_B" in signals

        # Should detect anomalies in both signals
        for signal in signals:
            signal_result = result[result["signal"] == signal]
            assert signal_result["is_anomaly"].sum() > 0

    def test_zscore_empty_df(self, empty_df_long: pd.DataFrame) -> None:
        """Test Z-Score with empty dataframe."""
        result = detect_zscore(empty_df_long)
        assert result.empty
        expected_cols = [
            "series_id",
            "signal",
            "timestamp_start",
            "timestamp_end",
            "score",
            "is_anomaly",
        ]
        assert list(result.columns) == expected_cols

    def test_zscore_constant_values(self) -> None:
        """Test Z-Score with constant values (std=0)."""
        df = pd.DataFrame(
            {
                "series_id": ["series_1"] * 5,
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
                "signal": ["sensor_A"] * 5,
                "value": [5.0, 5.0, 5.0, 5.0, 5.0],
            }
        )

        result = detect_zscore(df)

        # All scores should be 0 (no anomalies)
        assert (result["score"] == 0.0).all()
        assert not result["is_anomaly"].any()

    def test_zscore_output_schema(self, simple_df_long: pd.DataFrame) -> None:
        """Test that output has the expected schema."""
        result = detect_zscore(simple_df_long)

        expected_columns = [
            "series_id",
            "signal",
            "timestamp_start",
            "timestamp_end",
            "score",
            "is_anomaly",
        ]
        assert list(result.columns) == expected_columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_start"])
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_end"])
        assert pd.api.types.is_numeric_dtype(result["score"])
        assert result["is_anomaly"].dtype == bool


class TestIQR:
    """Tests for IQR anomaly detection."""

    def test_iqr_detects_outlier(self, simple_df_long: pd.DataFrame) -> None:
        """Test that IQR detects the obvious outlier."""
        result = detect_iqr(simple_df_long, multiplier=1.5)

        assert not result.empty
        assert "is_anomaly" in result.columns
        assert "score" in result.columns

        # The outlier (100.0) should be detected
        anomalies = result[result["is_anomaly"]]
        assert len(anomalies) > 0

    def test_iqr_multiplier_parameter(self, simple_df_long: pd.DataFrame) -> None:
        """Test that different multipliers affect anomaly detection."""
        result_strict = detect_iqr(simple_df_long, multiplier=3.0)
        result_lenient = detect_iqr(simple_df_long, multiplier=0.5)

        n_anomalies_strict = result_strict["is_anomaly"].sum()
        n_anomalies_lenient = result_lenient["is_anomaly"].sum()

        # Lenient multiplier should detect more anomalies
        assert n_anomalies_lenient >= n_anomalies_strict

    def test_iqr_multi_signal(self, multi_signal_df_long: pd.DataFrame) -> None:
        """Test IQR on multiple signals."""
        result = detect_iqr(multi_signal_df_long, multiplier=1.5)

        # Should have results for both signals
        signals = result["signal"].unique()
        assert len(signals) == 2

        # Should detect anomalies in both signals
        for signal in signals:
            signal_result = result[result["signal"] == signal]
            assert signal_result["is_anomaly"].sum() > 0

    def test_iqr_empty_df(self, empty_df_long: pd.DataFrame) -> None:
        """Test IQR with empty dataframe."""
        result = detect_iqr(empty_df_long)
        assert result.empty
        expected_cols = [
            "series_id",
            "signal",
            "timestamp_start",
            "timestamp_end",
            "score",
            "is_anomaly",
        ]
        assert list(result.columns) == expected_cols

    def test_iqr_normal_values_get_zero_score(self, simple_df_long: pd.DataFrame) -> None:
        """Test that values within bounds get zero score."""
        result = detect_iqr(simple_df_long, multiplier=1.5)

        # Most normal values should have score = 0
        normal_values = result[~result["is_anomaly"]]
        assert (normal_values["score"] == 0.0).all()

    def test_iqr_output_schema(self, simple_df_long: pd.DataFrame) -> None:
        """Test that output has the expected schema."""
        result = detect_iqr(simple_df_long)

        expected_columns = [
            "series_id",
            "signal",
            "timestamp_start",
            "timestamp_end",
            "score",
            "is_anomaly",
        ]
        assert list(result.columns) == expected_columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_start"])
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_end"])
        assert pd.api.types.is_numeric_dtype(result["score"])
        assert result["is_anomaly"].dtype == bool


class TestIsolationForest:
    """Tests for Isolation Forest anomaly detection."""

    def test_isolation_forest_detects_outlier(self, simple_df_long: pd.DataFrame) -> None:
        """Test that Isolation Forest detects the obvious outlier."""
        result = detect_isolation_forest(simple_df_long, contamination=0.1, random_state=42)

        assert not result.empty
        assert "is_anomaly" in result.columns
        assert "score" in result.columns

        # Should detect at least one anomaly
        anomalies = result[result["is_anomaly"]]
        assert len(anomalies) > 0

    def test_isolation_forest_contamination_parameter(self, simple_df_long: pd.DataFrame) -> None:
        """Test that contamination parameter affects anomaly detection."""
        result_low = detect_isolation_forest(simple_df_long, contamination=0.05, random_state=42)
        result_high = detect_isolation_forest(simple_df_long, contamination=0.3, random_state=42)

        n_anomalies_low = result_low["is_anomaly"].sum()
        n_anomalies_high = result_high["is_anomaly"].sum()

        # Higher contamination should detect more anomalies
        assert n_anomalies_high >= n_anomalies_low

    def test_isolation_forest_multi_signal(self, multi_signal_df_long: pd.DataFrame) -> None:
        """Test Isolation Forest on multiple signals."""
        result = detect_isolation_forest(multi_signal_df_long, contamination=0.1, random_state=42)

        # Should have results for both signals
        signals = result["signal"].unique()
        assert len(signals) == 2

    def test_isolation_forest_empty_df(self, empty_df_long: pd.DataFrame) -> None:
        """Test Isolation Forest with empty dataframe."""
        result = detect_isolation_forest(empty_df_long)
        assert result.empty
        expected_cols = [
            "series_id",
            "signal",
            "timestamp_start",
            "timestamp_end",
            "score",
            "is_anomaly",
        ]
        assert list(result.columns) == expected_cols

    def test_isolation_forest_single_value(self) -> None:
        """Test Isolation Forest with single value per signal."""
        df = pd.DataFrame(
            {
                "series_id": ["series_1"],
                "timestamp": pd.date_range("2024-01-01", periods=1, freq="h"),
                "signal": ["sensor_A"],
                "value": [5.0],
            }
        )

        result = detect_isolation_forest(df)

        # Should handle gracefully (mark as not anomalous)
        assert len(result) == 1
        assert not result["is_anomaly"].iloc[0]

    def test_isolation_forest_reproducibility(self, simple_df_long: pd.DataFrame) -> None:
        """Test that results are reproducible with same random_state."""
        result1 = detect_isolation_forest(simple_df_long, random_state=42)
        result2 = detect_isolation_forest(simple_df_long, random_state=42)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_isolation_forest_output_schema(self, simple_df_long: pd.DataFrame) -> None:
        """Test that output has the expected schema."""
        result = detect_isolation_forest(simple_df_long)

        expected_columns = [
            "series_id",
            "signal",
            "timestamp_start",
            "timestamp_end",
            "score",
            "is_anomaly",
        ]
        assert list(result.columns) == expected_columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_start"])
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_end"])
        assert pd.api.types.is_numeric_dtype(result["score"])
        assert result["is_anomaly"].dtype == bool

    def test_isolation_forest_scores_normalized(self, simple_df_long: pd.DataFrame) -> None:
        """Test that scores are normalized to [0, 1] range."""
        result = detect_isolation_forest(simple_df_long)

        assert (result["score"] >= 0.0).all()
        assert (result["score"] <= 1.0).all()


class TestAlgorithmsComparison:
    """Tests comparing behavior across algorithms."""

    def test_all_algorithms_detect_obvious_outlier(self, simple_df_long: pd.DataFrame) -> None:
        """Test that all algorithms detect the obvious outlier (100.0)."""
        result_zscore = detect_zscore(simple_df_long)
        result_iqr = detect_iqr(simple_df_long, multiplier=1.5)
        result_iso = detect_isolation_forest(simple_df_long, contamination=0.2, random_state=42)

        # All should detect at least one anomaly
        assert result_zscore["is_anomaly"].sum() > 0
        assert result_iqr["is_anomaly"].sum() > 0
        assert result_iso["is_anomaly"].sum() > 0

    def test_all_algorithms_consistent_schema(self, simple_df_long: pd.DataFrame) -> None:
        """Test that all algorithms return consistent schema."""
        result_zscore = detect_zscore(simple_df_long)
        result_iqr = detect_iqr(simple_df_long)
        result_iso = detect_isolation_forest(simple_df_long)

        expected_columns = [
            "series_id",
            "signal",
            "timestamp_start",
            "timestamp_end",
            "score",
            "is_anomaly",
        ]

        assert list(result_zscore.columns) == expected_columns
        assert list(result_iqr.columns) == expected_columns
        assert list(result_iso.columns) == expected_columns

    def test_all_algorithms_handle_empty_input(self, empty_df_long: pd.DataFrame) -> None:
        """Test that all algorithms handle empty input gracefully."""
        result_zscore = detect_zscore(empty_df_long)
        result_iqr = detect_iqr(empty_df_long)
        result_iso = detect_isolation_forest(empty_df_long)

        assert result_zscore.empty
        assert result_iqr.empty
        assert result_iso.empty
