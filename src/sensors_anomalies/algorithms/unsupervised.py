"""
Unsupervised anomaly detection algorithms.

All algorithms work per-sensor: each signal is analyzed independently.
Each algorithm returns a dataframe with columns:
- series_id: str
- signal: str
- timestamp_start: datetime
- timestamp_end: datetime
- score: float (anomaly score, higher = more anomalous)
- is_anomaly: bool (binary classification)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def detect_zscore(
    df_long: pd.DataFrame,
    threshold: float = 3.5,
    use_modified: bool = True,
) -> pd.DataFrame:  # pylint: disable=too-many-locals
    """
    Z-Score anomaly detection (per-sensor).

    Flags points where |z-score| > threshold. Works per signal independently.
    By default uses Modified Z-Score (based on median and MAD) which is more
    robust to outliers than standard Z-Score.

    Parameters
    ----------
    df_long : pd.DataFrame
        Normalized long-format dataframe with columns:
        series_id, timestamp, signal, value.
    threshold : float, optional
        Number of standard deviations from mean/median to flag as
        anomaly (default: 3.5).
    use_modified : bool, optional
        If True, use Modified Z-Score with median and MAD (more robust).
        If False, use standard Z-Score with mean and std (default: True).

    Returns
    -------
    pd.DataFrame
        Result dataframe with columns: series_id, signal, timestamp_start, timestamp_end, score, is_anomaly.
    """
    results = []

    # Group by series_id and signal
    for (series_id, signal), group in df_long.groupby(["series_id", "signal"], dropna=False):
        values = group["value"].to_numpy()
        timestamps = group["timestamp"].tolist()  # Use tolist() to preserve timezone info

        if use_modified:
            # Modified Z-Score: more robust to outliers
            median = np.median(values)
            mad = np.median(np.abs(values - median))

            if mad == 0:
                # MAD is zero - check if there's any variation at all
                # This handles cases where >50% of values are identical
                value_range = values.max() - values.min()

                if value_range > 0:
                    # Use absolute deviations from median as scores
                    # Normalize by the typical value (median) to get scale-independent scores
                    deviations = np.abs(values - median)
                    typical_value = np.abs(median) if median != 0 else 1.0
                    z_scores = deviations / typical_value
                else:
                    # All values are identical, no anomalies
                    z_scores = np.zeros(len(values))
            else:
                # Modified z-score formula: 0.6745 is the constant factor
                z_scores = np.abs(0.6745 * (values - median) / mad)
        else:
            # Standard Z-Score
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0.0

            # Use ternary operator: if std is 0, all values are the same (no anomalies)
            z_scores = np.zeros(len(values)) if std == 0 else np.abs((values - mean) / std)

        # Create result rows
        for _idx, (timestamp, z_score) in enumerate(zip(timestamps, z_scores, strict=True)):
            results.append(
                {
                    "series_id": series_id,
                    "signal": signal,
                    "timestamp_start": timestamp,
                    "timestamp_end": timestamp,
                    "score": float(z_score),
                    "is_anomaly": z_score > threshold,
                }
            )

    if not results:
        logger.warning("Z-Score detection: No data points found in input dataframe")
        return pd.DataFrame(columns=["series_id", "signal", "timestamp_start", "timestamp_end", "score", "is_anomaly"])

    result_df = pd.DataFrame(results)
    n_anomalies = result_df["is_anomaly"].sum()
    if n_anomalies == 0:
        logger.warning(
            "Z-Score detection: No anomalies detected (threshold=%s, %s points analyzed)",
            threshold,
            len(result_df),
        )
    else:
        logger.debug(
            "Z-Score detection: Found %s anomalies out of %s points (%.2f%%)",
            n_anomalies,
            len(result_df),
            n_anomalies / len(result_df) * 100,
        )

    return result_df


def detect_iqr(df_long: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:  # pylint: disable=too-many-locals
    """
    IQR (Interquartile Range) anomaly detection (per-sensor).

    Flags points outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    Works per signal independently.

    Parameters
    ----------
    df_long : pd.DataFrame
        Normalized long-format dataframe with columns: series_id, timestamp, signal, value.
    multiplier : float, optional
        IQR multiplier for outlier bounds (default: 1.5, Tukey's rule).

    Returns
    -------
    pd.DataFrame
        Result dataframe with columns: series_id, signal, timestamp_start, timestamp_end, score, is_anomaly.
    """
    results = []

    # Group by series_id and signal
    for (series_id, signal), group in df_long.groupby(["series_id", "signal"], dropna=False):
        values = group["value"].to_numpy()
        timestamps = group["timestamp"].tolist()  # Use tolist() to preserve timezone info

        # Calculate IQR bounds
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Calculate distance from nearest bound (normalized by IQR)
        # Score is 0 if within bounds, increases as point moves further out
        scores = np.zeros(len(values))
        for idx, val in enumerate(values):
            if val < lower_bound:
                scores[idx] = (lower_bound - val) / (iqr if iqr > 0 else 1.0)
            elif val > upper_bound:
                scores[idx] = (val - upper_bound) / (iqr if iqr > 0 else 1.0)

        # Create result rows
        for _idx, (timestamp, score) in enumerate(zip(timestamps, scores, strict=True)):
            results.append(
                {
                    "series_id": series_id,
                    "signal": signal,
                    "timestamp_start": timestamp,
                    "timestamp_end": timestamp,
                    "score": float(score),
                    "is_anomaly": score > 0,
                }
            )

    if not results:
        logger.warning("IQR detection: No data points found in input dataframe")
        return pd.DataFrame(columns=["series_id", "signal", "timestamp_start", "timestamp_end", "score", "is_anomaly"])

    result_df = pd.DataFrame(results)
    n_anomalies = result_df["is_anomaly"].sum()
    if n_anomalies == 0:
        logger.warning(
            "IQR detection: No anomalies detected (multiplier=%s, %s points analyzed)",
            multiplier,
            len(result_df),
        )
    else:
        logger.debug(
            "IQR detection: Found %s anomalies out of %s points (%.2f%%)",
            n_anomalies,
            len(result_df),
            n_anomalies / len(result_df) * 100,
        )

    return result_df


def detect_isolation_forest(
    df_long: pd.DataFrame,
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:  # pylint: disable=too-many-locals
    """
    Isolation Forest anomaly detection (per-sensor).

    Uses scikit-learn's Isolation Forest. Works per signal independently.

    Parameters
    ----------
    df_long : pd.DataFrame
        Normalized long-format dataframe with columns: series_id, timestamp, signal, value.
    contamination : float, optional
        Expected proportion of anomalies (default: 0.1 = 10%).
    n_estimators : int, optional
        Number of trees in the forest (default: 100).
    random_state : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    pd.DataFrame
        Result dataframe with columns: series_id, signal, timestamp_start, timestamp_end, score, is_anomaly.
    """
    results = []

    # Group by series_id and signal
    for (series_id, signal), group in df_long.groupby(["series_id", "signal"], dropna=False):
        values = group["value"].to_numpy().reshape(-1, 1)
        timestamps = group["timestamp"].tolist()  # Use tolist() to preserve timezone info

        # Need at least 2 samples for Isolation Forest
        if len(values) < 2:
            logger.warning(
                "Isolation Forest: Signal '%s' (series '%s') has only %s sample(s), marking all as normal",
                signal,
                series_id,
                len(values),
            )
            # Too few samples, mark all as normal
            for timestamp in timestamps:
                results.append(
                    {
                        "series_id": series_id,
                        "signal": signal,
                        "timestamp_start": timestamp,
                        "timestamp_end": timestamp,
                        "score": 0.0,
                        "is_anomaly": False,
                    }
                )
            continue

        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        predictions = iso_forest.fit_predict(values)  # 1 = normal, -1 = anomaly
        anomaly_scores = iso_forest.score_samples(values)  # More negative = more anomalous

        # Convert scores to positive scale (higher = more anomalous)
        # Normalize to [0, 1] range approximately
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        if max_score > min_score:
            normalized_scores = (max_score - anomaly_scores) / (max_score - min_score)
        else:
            normalized_scores = np.zeros(len(anomaly_scores))

        # Create result rows
        for _idx, (timestamp, pred, score) in enumerate(zip(timestamps, predictions, normalized_scores, strict=True)):
            results.append(
                {
                    "series_id": series_id,
                    "signal": signal,
                    "timestamp_start": timestamp,
                    "timestamp_end": timestamp,
                    "score": float(score),
                    "is_anomaly": pred == -1,
                }
            )

    if not results:
        logger.warning("Isolation Forest detection: No data points found in input dataframe")
        return pd.DataFrame(columns=["series_id", "signal", "timestamp_start", "timestamp_end", "score", "is_anomaly"])

    result_df = pd.DataFrame(results)
    n_anomalies = result_df["is_anomaly"].sum()
    if n_anomalies == 0:
        logger.warning(
            "Isolation Forest detection: No anomalies detected (contamination=%s, %s points analyzed)",
            contamination,
            len(result_df),
        )
    else:
        logger.debug(
            "Isolation Forest detection: Found %s anomalies out of %s points (%.2f%%)",
            n_anomalies,
            len(result_df),
            n_anomalies / len(result_df) * 100,
        )

    return result_df
