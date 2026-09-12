from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BASELINE_QUANTILE = 0.50
DEFAULT_PEAK_QUANTILE = 0.95
DEFAULT_MIN_DURATION_HOURS = 1.0


def detect_customer_peaks(
    group: pd.DataFrame,
    value_col: str,
    timestamp_col: str = "timestamp_local",
    baseline_quantile: float = DEFAULT_BASELINE_QUANTILE,
    peak_quantile: float = DEFAULT_PEAK_QUANTILE,
    min_duration_hours: float = DEFAULT_MIN_DURATION_HOURS,
) -> pd.DataFrame:
    """Detect peak events for ONE customer's series. Returns an event-level
    DataFrame (one row per detected event), without the id column attached
    (detect_all_peaks adds it)."""
    group = group.sort_values(timestamp_col)
    values = group[value_col].to_numpy(dtype=float)
    timestamps = group[timestamp_col].reset_index(drop=True)

    baseline = np.nanquantile(values, baseline_quantile)
    threshold = np.nanquantile(values, peak_quantile)
    is_peak = values >= threshold

    event_id = pd.Series(is_peak).ne(pd.Series(is_peak).shift()).cumsum().to_numpy()

    default_step = (
        timestamps.diff().dropna().median() if len(timestamps) > 1 else pd.Timedelta(hours=1)
    )

    rows = []
    for event in np.unique(event_id[is_peak]):
        mask = is_peak & (event_id == event)
        event_values = values[mask]
        event_times = timestamps[mask]
        if len(event_times) == 0:
            continue

        start = event_times.iloc[0]
        end_observed = event_times.iloc[-1]
        step = event_times.diff().dropna().median() if len(event_times) > 1 else default_step
        end = end_observed + step
        duration_hours = (end - start).total_seconds() / 3600

        if duration_hours < min_duration_hours:
            continue

        peak_value = event_values.max()
        peak_time = event_times.iloc[np.argmax(event_values)]

        rows.append({
            "peak_start": start,
            "peak_end": end,
            "duration_hours": duration_hours,
            "peak_time": peak_time,
            "peak_value": peak_value,
            "baseline": baseline,
            "threshold": threshold,
            "peak_ratio": peak_value / baseline if baseline != 0 else np.nan,
            "peak_excess": peak_value - baseline,
            "is_weekend": bool(start.dayofweek >= 5),
        })

    return pd.DataFrame(rows)


def detect_all_peaks(
    long_df: pd.DataFrame,
    value_col: str,
    id_col: str = "customer_id",
    timestamp_col: str = "timestamp_local",
    **kwargs,
) -> pd.DataFrame:
    """Run detect_customer_peaks for every customer in long_df. Returns one
    combined event-level table with id_col as the first column."""
    tables = []
    for cid, group in long_df.groupby(id_col, sort=False):
        detected = detect_customer_peaks(group, value_col, timestamp_col, **kwargs)
        if not detected.empty:
            detected.insert(0, id_col, cid)
            tables.append(detected)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def summarize_peaks(peak_events: pd.DataFrame, id_col: str = "customer_id") -> pd.DataFrame:
    """Aggregate event-level peaks into one row of peak-derived features per
    customer. Customers with zero detected events simply won't appear here -
    the caller should left-join this onto the full customer list."""
    if peak_events.empty:
        return pd.DataFrame(columns=[id_col])

    peak_events = peak_events.copy()
    peak_events["peak_hour"] = (
        peak_events["peak_start"].dt.hour + peak_events["peak_start"].dt.minute / 60
    )

    def _agg(group):
        start_hour = group["peak_hour"]
        return pd.Series({
            "peak_count": len(group),
            "max_peak": group["peak_value"].max(),
            "median_peak": group["peak_value"].median(),
            "mean_peak": group["peak_value"].mean(),
            "mean_peak_duration_hours": group["duration_hours"].mean(),
            "max_peak_duration_hours": group["duration_hours"].max(),
            "mean_peak_ratio": group["peak_ratio"].mean(),
            "max_peak_ratio": group["peak_ratio"].max(),
            "morning_peak_share": float(((start_hour >= 6) & (start_hour < 10)).mean()),
            "daytime_peak_share": float(((start_hour >= 10) & (start_hour < 17)).mean()),
            "evening_peak_share": float(((start_hour >= 17) & (start_hour < 22)).mean()),
            "weekday_peak_share": float((group["peak_start"].dt.dayofweek < 5).mean()),
        })

    return peak_events.groupby(id_col, sort=False).apply(_agg).reset_index()
