from __future__ import annotations

import numpy as np
import pandas as pd

import peaks as _peaks


def _interval_hours_for_group(group: pd.DataFrame, timestamp_col: str) -> float:
    """Median gap between consecutive readings, in hours - inferred from
    the data rather than hardcoded, so this works for any resolution."""
    ts = group[timestamp_col].sort_values()
    diffs = ts.diff().dropna()
    if len(diffs) == 0:
        return 1.0
    return diffs.median().total_seconds() / 3600


def basic_features(
    long_df: pd.DataFrame,
    value_col: str,
    id_col: str = "customer_id",
    value_type: str = "power",
    timestamp_col: str = "timestamp_local",
) -> pd.DataFrame:
    """value_type: 'power' (value_col is an instantaneous rate, e.g.
    power_kw - annual_consumption is computed by time-weighting each
    reading by its interval duration) or 'energy' (value_col already IS
    energy consumed during its interval, e.g. gas's energy_kwh -
    annual_consumption is a direct sum, no further scaling)."""
    if value_type not in ("power", "energy"):
        raise ValueError(f"value_type must be 'power' or 'energy', got {value_type!r}")

    def _per_customer(group):
        values = group[value_col].to_numpy(dtype=float)
        mean_value = values.mean()
        max_value = values.max()

        if value_type == "power":
            interval_hours = _interval_hours_for_group(group, timestamp_col)
            annual_energy = values.sum() * interval_hours
        else:
            annual_energy = values.sum()

        return pd.Series({
            "annual_consumption": annual_energy,
            "mean_consumption": mean_value,
            "median_consumption": np.median(values),
            "min_consumption": values.min(),
            "max_consumption": max_value,
            "std_consumption": values.std(),
            "p95_consumption": np.percentile(values, 95),
            "p99_consumption": np.percentile(values, 99),
            "coefficient_of_variation": values.std() / mean_value if mean_value != 0 else np.nan,
            "load_factor": mean_value / max_value if max_value != 0 else np.nan,
        })

    return long_df.groupby(id_col, sort=False).apply(_per_customer).reset_index()


def time_of_day_features(
    long_df: pd.DataFrame,
    value_col: str,
    id_col: str = "customer_id",
    timestamp_col: str = "timestamp_local",
) -> pd.DataFrame:
    def _per_customer(group):
        hour = group[timestamp_col].dt.hour + group[timestamp_col].dt.minute / 60
        total = group[value_col].sum()

        def share(mask):
            return group.loc[mask, value_col].sum() / total if total != 0 else np.nan

        weekday = group[timestamp_col].dt.dayofweek < 5
        return pd.Series({
            "night_share": share((hour >= 0) & (hour < 6)),
            "morning_share": share((hour >= 6) & (hour < 10)),
            "daytime_share": share((hour >= 10) & (hour < 17)),
            "evening_share": share((hour >= 17) & (hour < 22)),
            "weekday_share": share(weekday),
            "weekend_share": share(~weekday),
        })

    return long_df.groupby(id_col, sort=False).apply(_per_customer).reset_index()


def build_feature_matrix(
    long_df: pd.DataFrame,
    peak_events: pd.DataFrame,
    value_col: str,
    id_col: str = "customer_id",
    timestamp_col: str = "timestamp_local",
    value_type: str = "power",
) -> pd.DataFrame:
    """Combine basic + time-of-day + peak-derived features into the full
    matrix. Customers with zero detected peak events get NaN peak features
    (not dropped) - the caller decides how a downstream step (e.g. K-Means)
    should handle that (impute, or treat "no peaks" as informative).

    value_type: see basic_features - 'power' for electricity (power_kw),
    'energy' for gas (energy_kwh)."""
    basic = basic_features(long_df, value_col, id_col, value_type=value_type, timestamp_col=timestamp_col)
    time_feats = time_of_day_features(long_df, value_col, id_col, timestamp_col)
    peak_feats = (
        _peaks.summarize_peaks(peak_events, id_col) if peak_events is not None and not peak_events.empty
        else pd.DataFrame({id_col: basic[id_col]})
    )

    features = (
        basic.merge(time_feats, on=id_col, how="left")
             .merge(peak_feats, on=id_col, how="left")
    )
    return features.replace([np.inf, -np.inf], np.nan)
