"""Tests for period-based anomaly detection."""
# ruff: noqa: S101
# pylint: disable=import-error,redefined-outer-name,duplicate-code

from __future__ import annotations

import pandas as pd
import pytest

from sensors_anomalies.algorithms.periods import detect_anomaly_periods


@pytest.fixture
def point_results_simple() -> pd.DataFrame:
    """Create simple point-by-point detection results."""
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h")

    # Create pattern: 5 normal, 5 high-score (anomaly period), 5 normal, 5 high-score
    scores = [0.5] * 5 + [3.0] * 5 + [0.5] * 5 + [2.5] * 5

    return pd.DataFrame(
        {
            "series_id": ["s1"] * 20,
            "signal": ["sensor_A"] * 20,
            "timestamp_start": timestamps,
            "timestamp_end": timestamps,
            "score": scores,
            "is_anomaly": [s > 1.0 for s in scores],
            "algorithm": ["zscore"] * 20,
        }
    )


@pytest.fixture
def point_results_multi_algo() -> pd.DataFrame:
    """Create point results with multiple algorithms."""
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h")

    df1 = pd.DataFrame(
        {
            "series_id": ["s1"] * 10,
            "signal": ["sensor_A"] * 10,
            "timestamp_start": timestamps,
            "timestamp_end": timestamps,
            "score": [0.5] * 3 + [3.0] * 4 + [0.5] * 3,
            "is_anomaly": [False] * 3 + [True] * 4 + [False] * 3,
            "algorithm": ["zscore"] * 10,
        }
    )

    df2 = pd.DataFrame(
        {
            "series_id": ["s1"] * 10,
            "signal": ["sensor_A"] * 10,
            "timestamp_start": timestamps,
            "timestamp_end": timestamps,
            "score": [0.5] * 2 + [2.5] * 5 + [0.5] * 3,
            "is_anomaly": [False] * 2 + [True] * 5 + [False] * 3,
            "algorithm": ["iqr"] * 10,
        }
    )

    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def empty_point_results() -> pd.DataFrame:
    """Create empty point results dataframe."""
    return pd.DataFrame(
        columns=[
            "series_id", "signal", "timestamp_start", "timestamp_end",
            "score", "is_anomaly", "algorithm"
        ]
    )


class TestDetectAnomalyPeriods:
    """Tests for period detection functionality."""

    def test_basic_period_detection(self, point_results_simple: pd.DataFrame) -> None:
        """Test basic period detection with clear anomaly periods."""
        result = detect_anomaly_periods(point_results_simple, k=3, n=5, score_threshold=2.0)

        assert not result.empty
        assert "timestamp_start" in result.columns
        assert "timestamp_end" in result.columns
        assert "n_points" in result.columns
        assert "mean_score" in result.columns
        assert "max_score" in result.columns

        # Should detect 2 periods (indices 5-9 and 15-19)
        assert len(result) == 2

        # Check period properties
        assert all(result["n_points"] == 5)
        assert all(result["mean_score"] >= 2.0)

    def test_k_parameter_affects_detection(self, point_results_simple: pd.DataFrame) -> None:
        """Test that K parameter affects number of detected periods."""
        result_strict = detect_anomaly_periods(
            point_results_simple, k=5, n=5, score_threshold=2.0
        )
        result_lenient = detect_anomaly_periods(point_results_simple, k=2, n=5, score_threshold=2.0)

        # Lenient (lower K) should detect more or equal periods
        assert len(result_lenient) >= len(result_strict)

    def test_threshold_parameter_affects_detection(
        self, point_results_simple: pd.DataFrame
    ) -> None:
        """Test that score threshold affects period detection."""
        result_low = detect_anomaly_periods(
            point_results_simple, k=3, n=5, score_threshold=1.0
        )
        result_high = detect_anomaly_periods(
            point_results_simple, k=3, n=5, score_threshold=4.0
        )

        # Lower threshold should detect more or equal periods
        assert len(result_low) >= len(result_high)

    def test_multi_algorithm_detection(self, point_results_multi_algo: pd.DataFrame) -> None:
        """Test period detection with multiple algorithms."""
        result = detect_anomaly_periods(point_results_multi_algo, k=3, n=5, score_threshold=2.0)

        # Should have results from both algorithms
        algorithms = result["algorithm"].unique()
        assert len(algorithms) == 2
        assert "zscore" in algorithms
        assert "iqr" in algorithms

        # Each algorithm should have at least one period
        for algo in algorithms:
            algo_periods = result[result["algorithm"] == algo]
            assert len(algo_periods) > 0

    def test_empty_input(self, empty_point_results: pd.DataFrame) -> None:
        """Test period detection with empty input."""
        result = detect_anomaly_periods(empty_point_results, k=3, n=5, score_threshold=2.0)

        assert result.empty
        expected_cols = [
            "series_id",
            "signal",
            "algorithm",
            "timestamp_start",
            "timestamp_end",
            "n_points",
            "mean_score",
            "max_score",
        ]
        assert list(result.columns) == expected_cols

    def test_invalid_k_n_parameters(self, point_results_simple: pd.DataFrame) -> None:
        """Test that invalid K/N parameters raise errors."""
        # K > N should raise error
        with pytest.raises(ValueError, match="k=.* must be <= n="):
            detect_anomaly_periods(point_results_simple, k=6, n=5, score_threshold=2.0)

        # K < 1 should raise error
        with pytest.raises(ValueError, match="k=.* and n=.* must be >= 1"):
            detect_anomaly_periods(point_results_simple, k=0, n=5, score_threshold=2.0)

        # N < 1 should raise error
        with pytest.raises(ValueError, match="k=.* and n=.* must be >= 1"):
            detect_anomaly_periods(point_results_simple, k=3, n=0, score_threshold=2.0)

    def test_missing_required_columns(self) -> None:
        """Test that missing required columns raises error."""
        incomplete_df = pd.DataFrame(
            {
                "series_id": ["s1"] * 5,
                "signal": ["sensor_A"] * 5,
                # Missing other required columns
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            detect_anomaly_periods(incomplete_df, k=2, n=3, score_threshold=1.0)

    def test_output_schema(self, point_results_simple: pd.DataFrame) -> None:
        """Test that output has expected schema and data types."""
        result = detect_anomaly_periods(point_results_simple, k=3, n=5, score_threshold=2.0)

        expected_columns = [
            "series_id",
            "signal",
            "algorithm",
            "timestamp_start",
            "timestamp_end",
            "n_points",
            "mean_score",
            "max_score",
        ]
        assert list(result.columns) == expected_columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_start"])
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_end"])
        assert pd.api.types.is_numeric_dtype(result["n_points"])
        assert pd.api.types.is_numeric_dtype(result["mean_score"])
        assert pd.api.types.is_numeric_dtype(result["max_score"])

    def test_period_end_after_start(self, point_results_simple: pd.DataFrame) -> None:
        """Test that period end timestamps are after start timestamps."""
        result = detect_anomaly_periods(point_results_simple, k=3, n=5, score_threshold=2.0)

        if not result.empty:
            assert all(result["timestamp_end"] >= result["timestamp_start"])

    def test_consecutive_periods_merged(self) -> None:
        """Test that consecutive anomaly points form continuous periods."""
        # Create data with long consecutive anomaly sequence
        timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "series_id": ["s1"] * 10,
                "signal": ["sensor_A"] * 10,
                "timestamp_start": timestamps,
                "timestamp_end": timestamps,
                "score": [3.0] * 10,  # All high scores
                "is_anomaly": [True] * 10,
                "algorithm": ["zscore"] * 10,
            }
        )

        result = detect_anomaly_periods(df, k=2, n=3, score_threshold=2.0)

        # Should create one continuous period, not multiple separate ones
        assert len(result) == 1
        assert result.iloc[0]["n_points"] == 10

    def test_sliding_window_behavior(self) -> None:
        """Test sliding window detection behavior."""
        # Pattern: 2 normal, 3 high, 2 normal, 3 high
        timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
        scores = [0.5, 0.5, 3.0, 3.0, 3.0, 0.5, 0.5, 3.0, 3.0, 3.0]

        df = pd.DataFrame(
            {
                "series_id": ["s1"] * 10,
                "signal": ["sensor_A"] * 10,
                "timestamp_start": timestamps,
                "timestamp_end": timestamps,
                "score": scores,
                "is_anomaly": [s > 1.0 for s in scores],
                "algorithm": ["zscore"] * 10,
            }
        )

        # With k=2, n=3: should detect both sequences of 3 high scores
        result = detect_anomaly_periods(df, k=2, n=3, score_threshold=2.0)

        assert len(result) == 2  # Two separate periods

    def test_no_periods_detected(self) -> None:
        """Test when no periods meet criteria."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "series_id": ["s1"] * 10,
                "signal": ["sensor_A"] * 10,
                "timestamp_start": timestamps,
                "timestamp_end": timestamps,
                "score": [0.5] * 10,  # All low scores
                "is_anomaly": [False] * 10,
                "algorithm": ["zscore"] * 10,
            }
        )

        result = detect_anomaly_periods(df, k=3, n=5, score_threshold=2.0)

        assert result.empty

    def test_per_signal_analysis(self) -> None:
        """Test that period detection is done per-signal independently."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="h")

        # Create data for two signals with different anomaly patterns
        df = pd.DataFrame(
            {
                "series_id": ["s1"] * 20,
                "signal": ["sensor_A"] * 10 + ["sensor_B"] * 10,
                "timestamp_start": timestamps.tolist() * 2,
                "timestamp_end": timestamps.tolist() * 2,
                "score": [3.0] * 5 + [0.5] * 5 + [0.5] * 5 + [3.0] * 5,
                "is_anomaly": [True] * 5 + [False] * 5 + [False] * 5 + [True] * 5,
                "algorithm": ["zscore"] * 20,
            }
        )

        result = detect_anomaly_periods(df, k=3, n=5, score_threshold=2.0)

        # Should have 2 periods total (one per signal)
        assert len(result) == 2

        # Check each signal has its own period
        signals = result["signal"].unique()
        assert len(signals) == 2
        assert "sensor_A" in signals
        assert "sensor_B" in signals
