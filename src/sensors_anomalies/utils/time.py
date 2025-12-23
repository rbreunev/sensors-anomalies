"""Time handling utilities."""

from __future__ import annotations

import pandas as pd


def ensure_utc(ts: pd.Series) -> pd.Series:
    """
    Ensure a pandas datetime series is timezone-aware in UTC.

    Parameters
    ----------
    ts : pd.Series
        Datetime series to normalize.

    Returns
    -------
    pd.Series
        Timezone-aware series in UTC.

    Raises
    ------
    TypeError
        If the input series is not of a datetime dtype.
    """
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise TypeError("timestamp must be datetime dtype")

    # ts.dt.tz can be None when timestamps are tz-naive
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")  # type: ignore[no-any-return]
    return ts.dt.tz_convert("UTC")  # type: ignore[no-any-return]
