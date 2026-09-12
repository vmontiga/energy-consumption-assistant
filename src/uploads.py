"""
uploads.py
----------
Parses a user-uploaded consumption file (CSV or Excel) into the canonical
long-format schema the rest of the pipeline expects:
    timestamp_local | timestamp_utc | customer_id | power_kw

Expected input: the exact format the single-customer generator produces
(src/synthetic_generator.py, saved by notebook 00's "Single Customer
Generation for Testing Purposes" section) - already-clean, single-customer,
already has timestamp_local/timestamp_utc columns. A real customer's own
export would need to match this same shape.

This is deliberately NOT a general raw-file parser (German number formats,
multi-customer sheets, DST correction for genuinely raw meter exports) -
that concern was handled by a since-removed real-data pipeline
(src/io_loading.py + src/time_cleaning.py's standardize_timestamps, see
project history). If raw-file support is needed again later, that's a
separate, larger piece of work - not what this module does.
"""

from __future__ import annotations

import io

import pandas as pd

REQUIRED_COLUMNS = {"timestamp_utc", "customer_id", "power_kw"}
TIMEZONE = "Europe/Berlin"


class UploadError(ValueError):
    """Raised when an uploaded file can't be parsed into the expected
    schema - the message is written to be shown directly to the app user."""


def parse_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV or Excel file into the canonical electricity
    long-format DataFrame (timestamp_local | timestamp_utc | customer_id |
    power_kw | source_file). Raises UploadError with a clear, user-facing
    message on any problem."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    buffer = io.BytesIO(file_bytes)

    if suffix == "csv":
        try:
            raw = pd.read_csv(buffer)
        except Exception as exc:  # noqa: BLE001
            raise UploadError(f"Could not read this as a CSV file: {exc}") from exc
    elif suffix in ("xlsx", "xls"):
        try:
            raw = pd.read_excel(buffer)
        except Exception as exc:  # noqa: BLE001
            raise UploadError(f"Could not read this as an Excel file: {exc}") from exc
    else:
        raise UploadError(
            f"Unsupported file type: '.{suffix}'. Please upload a .csv or .xlsx file."
        )

    if raw.empty:
        raise UploadError("The uploaded file contains no data rows.")

    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise UploadError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Expected at least: {', '.join(sorted(REQUIRED_COLUMNS))}."
        )

    # Robustly reconstruct tz-aware timestamps: Europe/Berlin timestamps carry
    # DIFFERENT UTC offsets across a year (+01:00 winter, +02:00 summer, DST),
    # so a naive datetime parse of timestamp_utc's string form can misread it -
    # parsing with utc=True first (honoring each row's own explicit offset) and
    # deriving timestamp_local from that is the reliable way (see
    # time_cleaning.read_long_csv, which this mirrors).
    try:
        timestamp_utc = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise UploadError(f"Could not parse the 'timestamp_utc' column as dates/times: {exc}") from exc

    power_kw = pd.to_numeric(raw["power_kw"], errors="coerce")
    if power_kw.isna().any():
        n_bad = int(power_kw.isna().sum())
        raise UploadError(
            f"{n_bad} row(s) in 'power_kw' are not valid numbers - please check the uploaded file."
        )

    customer_ids = raw["customer_id"].dropna().unique()
    if len(customer_ids) != 1:
        preview = ", ".join(str(c) for c in customer_ids[:5])
        raise UploadError(
            f"Expected exactly one customer in the uploaded file, found {len(customer_ids)}: "
            f"{preview}{', ...' if len(customer_ids) > 5 else ''}."
        )

    df = pd.DataFrame({
        "timestamp_utc": timestamp_utc,
        "timestamp_local": timestamp_utc.dt.tz_convert(TIMEZONE),
        "customer_id": customer_ids[0],
        "power_kw": power_kw,
    })
    df["source_file"] = filename

    return df.sort_values("timestamp_utc").reset_index(drop=True)
