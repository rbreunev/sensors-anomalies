"""
Period-based anomaly detection.

This module provides functionality to detect anomaly periods from point-by-point
anomaly detection results. A period is defined as a contiguous time range where
K out of N consecutive points exceed a score threshold.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_anomaly_periods(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    point_results: pd.DataFrame,
    k: int = 3,
    n: int = 5,
    score_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Detect anomaly periods from point-by-point detection results.

    Uses a sliding window approach: for each point that exceeds the score threshold,
    checks if there are K or more anomalous points within a window of size N centered
    around it. If so, the point is marked as part of an anomaly period. Consecutive
    marked points are merged into single periods.

    Parameters
    ----------
    point_results : pd.DataFrame
        Point-by-point detection results with columns:
        series_id, signal, timestamp_start, timestamp_end, score, is_anomaly, algorithm
    k : int, optional
        Minimum number of anomalous points in window (default: 3)
    n : int, optional
        Window size for checking anomalies (default: 5)
    score_threshold : float, optional
        Minimum score to consider a point anomalous (default: 1.0)

    Returns
    -------
    pd.DataFrame
        Period-based results with columns:
        series_id, signal, algorithm, timestamp_start, timestamp_end,
        n_points, mean_score, max_score

    Raises
    ------
    ValueError
        If k > n or if required columns are missing
    """
    # Validate parameters
    if k < 1 or n < 1:
        raise ValueError(f"k={k} and n={n} must be >= 1")

    if k > n:
        raise ValueError(f"k={k} must be <= n={n}")

    # Check required columns
    required_cols = ["series_id", "signal", "timestamp_start", "score", "algorithm"]
    missing_cols = set(required_cols) - set(point_results.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if point_results.empty:
        logger.warning("Period detection: Empty input dataframe")
        return pd.DataFrame(
            columns=[
                "series_id",
                "signal",
                "algorithm",
                "timestamp_start",
                "timestamp_end",
                "n_points",
                "mean_score",
                "max_score",
            ]
        )

    periods = []

    # Process each (series_id, signal, algorithm) group independently
    for (series_id, signal, algorithm), group in point_results.groupby(
        ["series_id", "signal", "algorithm"], dropna=False
    ):
        # Sort by timestamp to ensure correct order
        group = group.sort_values("timestamp_start").reset_index(drop=True)

        scores = group["score"].to_numpy()
        timestamps = group["timestamp_start"].to_numpy()

        # For each point, check if K out of next N points exceed threshold
        # This determines if the point is part of an anomaly period
        is_in_period = np.zeros(len(scores), dtype=bool)

        for i, _score in enumerate(scores):
            # Only check points that exceed threshold
            if scores[i] < score_threshold:
                continue

            # Look at a window centered around this point (including current position)
            # This ensures edge points are detected if they're part of a cluster
            window_start = max(0, i - n + 1)
            window_end = min(i + n, len(scores))
            window_scores = scores[window_start:window_end]

            # Count how many points in this window exceed threshold
            n_above = np.sum(window_scores >= score_threshold)

            # If K or more points exceed threshold in the window,
            # mark this point as in a period
            if n_above >= k:
                is_in_period[i] = True

        # Group consecutive True values into periods
        if not np.any(is_in_period):
            continue

        # Find runs of consecutive points marked as in_period
        period_start_idx = None
        for i, in_period_flag in enumerate(is_in_period):
            if in_period_flag and period_start_idx is None:
                # Start new period
                period_start_idx = i
            elif not in_period_flag and period_start_idx is not None:
                # End current period at previous index
                period_end_idx = i - 1

                # Get indices of points that exceed threshold in this period
                period_indices = range(period_start_idx, period_end_idx + 1)
                anomalous_indices = [
                    idx for idx in period_indices
                    if scores[idx] >= score_threshold
                ]

                if anomalous_indices:  # Should always be true, but safety check
                    period_scores = scores[anomalous_indices]
                    # Use timestamps of first and last anomalous points
                    period_timestamp_start = timestamps[anomalous_indices[0]]
                    period_timestamp_end = timestamps[anomalous_indices[-1]]

                    periods.append(
                        {
                            "series_id": series_id,
                            "signal": signal,
                            "algorithm": algorithm,
                            "timestamp_start": period_timestamp_start,
                            "timestamp_end": period_timestamp_end,
                            "n_points": len(period_scores),
                            "mean_score": float(np.mean(period_scores)),
                            "max_score": float(np.max(period_scores)),
                        }
                    )

                period_start_idx = None

        # Handle case where period extends to end of data
        if period_start_idx is not None:
            period_indices = range(period_start_idx, len(scores))
            anomalous_indices = [
                idx for idx in period_indices
                if scores[idx] >= score_threshold
            ]

            if anomalous_indices:  # Should always be true, but safety check
                period_scores = scores[anomalous_indices]
                # Use timestamps of first and last anomalous points
                period_timestamp_start = timestamps[anomalous_indices[0]]
                period_timestamp_end = timestamps[anomalous_indices[-1]]

                periods.append(
                    {
                        "series_id": series_id,
                        "signal": signal,
                        "algorithm": algorithm,
                        "timestamp_start": period_timestamp_start,
                        "timestamp_end": period_timestamp_end,
                        "n_points": len(period_scores),
                        "mean_score": float(np.mean(period_scores)),
                        "max_score": float(np.max(period_scores)),
                    }
                )

    if not periods:
        logger.info(
            "Period detection: No periods detected (k=%s, n=%s, threshold=%s, %s points analyzed)",
            k,
            n,
            score_threshold,
            len(point_results),
        )
        return pd.DataFrame(
            columns=[
                "series_id",
                "signal",
                "algorithm",
                "timestamp_start",
                "timestamp_end",
                "n_points",
                "mean_score",
                "max_score",
            ]
        )

    result_df = pd.DataFrame(periods)

    # Ensure timestamp columns are datetime
    result_df["timestamp_start"] = pd.to_datetime(result_df["timestamp_start"])
    result_df["timestamp_end"] = pd.to_datetime(result_df["timestamp_end"])

    logger.info(
        "Period detection: Found %s periods from %s points (k=%s, n=%s, threshold=%s)",
        len(result_df),
        len(point_results),
        k,
        n,
        score_threshold,
    )

    return result_df
