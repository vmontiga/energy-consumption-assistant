from __future__ import annotations

import pandas as pd

TIMEZONE = "Europe/Berlin"


def validate_continuity(df: pd.DataFrame, id_col: str, expected_freq: str) -> pd.DataFrame:
    """For each customer, check the timestamp_utc series is duplicate-free
    and has no gaps at the given frequency. Returns a per-customer summary
    for review - does not raise, since real (non-DST-related) gaps from
    meter downtime are a data-quality finding to surface, not an error to
    hide."""
    rows = []
    for cid, g in df.groupby(id_col):
        g = g.sort_values("timestamp_utc")
        n_dupes = int(g["timestamp_utc"].duplicated().sum())
        full_range = pd.date_range(g["timestamp_utc"].min(), g["timestamp_utc"].max(), freq=expected_freq)
        missing = full_range.difference(g["timestamp_utc"])
        rows.append({
            id_col: cid,
            "n_rows": len(g),
            "n_duplicate_utc_timestamps": n_dupes,
            "n_missing_intervals": len(missing),
            "first_missing": missing.min() if len(missing) else pd.NaT,
        })
    return pd.DataFrame(rows)


def parse_local_timestamp(series: pd.Series) -> pd.Series:
    """Robustly parse a column of Europe/Berlin tz-aware timestamps that
    were saved to CSV (and so may show different literal UTC offsets on
    different rows, e.g. '+01:00' in winter vs '+02:00' in summer) back
    into a proper tz-aware dtype. Each row's offset is explicit in its ISO
    string, so parsing with utc=True first (which honors that per-row
    offset) and then converting is reliable regardless of the mix; asking
    pandas to infer a single tz-aware dtype directly from mixed-offset
    strings is what fails. Use this for any saved local-time column that
    doesn't have a paired *_utc column to fall back on (e.g. peak_start /
    peak_end / peak_time in the peak-events tables) - see read_long_csv for
    the long-format electricity/gas tables, which do have a UTC column."""
    return pd.to_datetime(series, utc=True).dt.tz_convert(TIMEZONE)


def read_long_csv(path) -> pd.DataFrame:
    """Load a long-format electricity/gas CSV saved by this project, safely
    reconstructing tz-aware timestamps.

    Why this exists: Europe/Berlin timestamps carry DIFFERENT UTC offsets
    across a year (+01:00 in winter, +02:00 in summer, due to DST). When
    such a tz-aware column is saved to CSV and reloaded via the common
    pattern `pd.read_csv(path, parse_dates=[...])`, pandas cannot reliably
    unify the mixed offsets into one tz-aware dtype and silently leaves the
    column as plain strings instead - which then breaks downstream on the
    first `.diff()` or similar datetime operation, often far from the
    actual cause. `timestamp_utc` alone has a uniform offset (+00:00) and
    parses safely with `utc=True`; `timestamp_local` is then re-derived
    from it via `tz_convert`, sidestepping the mixed-offset parsing issue
    entirely rather than fighting it.

    Always use this (not a raw pd.read_csv) for any CSV containing
    timestamp_local/timestamp_utc columns.
    """
    df = pd.read_csv(path)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(TIMEZONE)
    return df
