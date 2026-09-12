from __future__ import annotations

from pathlib import Path

import pandas as pd
import pdfplumber

# The single tariff row used for all customers in this project.
# Expandable later simply by selecting a different row from the
# already-extracted, already-normalized table - no PDF re-parsing needed.
DEFAULT_WESTNETZ_VOLTAGE_LEVEL = "Mittelspannung mit Umspannung auf Niederspannung"

_WESTNETZ_VOLTAGE_ORDER = [
    "Höchstspannung mit Umspannung auf Hochspannung",
    "Hochspannung",
    "Hochspannung mit Umspannung auf Mittelspannung",
    "Mittelspannung",
    "Mittelspannung mit Umspannung auf Niederspannung",
    "Niederspannung",
]

_AMPRION_VOLTAGE_ORDER = [
    "Höchstspannung",
    "Höchstspannung mit Umspannung",
]


def _parse_de_number(text: str) -> float:
    return float(text.strip().replace(".", "").replace(",", "."))


def extract_westnetz_annual_tariffs(pdf_path: str | Path) -> pd.DataFrame:
    """Parse Preisblatt 1 (Jahresleistungspreissystem) - page 1 of the
    Westnetz price sheet. Returns one row per voltage level:
    voltage_level | low_util_leistungspreis_eur_kwa | low_util_arbeitspreis_ct_kwh |
    high_util_leistungspreis_eur_kwa | high_util_arbeitspreis_ct_kwh
    ('low_util' = <2500 h/a, 'high_util' = >=2500 h/a annual utilization hours)
    """
    with pdfplumber.open(pdf_path) as pdf:
        tables = pdf.pages[0].extract_tables()

    data_rows = []
    for table in tables:
        for row in table:
            if row and row[0] in _WESTNETZ_VOLTAGE_ORDER:
                data_rows.append(row)

    if len(data_rows) != len(_WESTNETZ_VOLTAGE_ORDER):
        raise ValueError(
            f"Expected {len(_WESTNETZ_VOLTAGE_ORDER)} Westnetz voltage-level rows, "
            f"found {len(data_rows)}. PDF layout may have changed - inspect page 1 manually."
        )

    records = []
    for row in data_rows:
        voltage_level, low_util_cell, high_util_cell = row[0], row[1], row[2]
        low_leistung, low_arbeit = low_util_cell.split()
        high_leistung, high_arbeit = high_util_cell.split()
        records.append({
            "voltage_level": voltage_level,
            "low_util_leistungspreis_eur_kwa": _parse_de_number(low_leistung),
            "low_util_arbeitspreis_ct_kwh": _parse_de_number(low_arbeit),
            "high_util_leistungspreis_eur_kwa": _parse_de_number(high_leistung),
            "high_util_arbeitspreis_ct_kwh": _parse_de_number(high_arbeit),
        })

    df = pd.DataFrame(records)
    df["voltage_level"] = pd.Categorical(df["voltage_level"], categories=_WESTNETZ_VOLTAGE_ORDER, ordered=True)
    return df.sort_values("voltage_level").reset_index(drop=True)


def extract_amprion_annual_tariffs(pdf_path: str | Path) -> pd.DataFrame:
    """Parse the Jahresleistungspreissystem table (page 1, first table) of
    the Amprion price sheet. Returns the same schema as
    extract_westnetz_annual_tariffs, with the 2 Amprion voltage categories."""
    with pdfplumber.open(pdf_path) as pdf:
        tables = pdf.pages[0].extract_tables()

    # The first table on the page is the annual demand-price system table.
    # Data rows are 4 already-separate numeric cells (unlike Westnetz, whose
    # cells are merged pairs) - but row labels aren't captured by pdfplumber
    # here (multi-line left-column labels get lost), so rows are matched
    # back to voltage levels by their fixed, known display order.
    table = tables[0]
    data_rows = [row for row in table if row[0] is not None and _is_numeric_de(row[0])]

    if len(data_rows) != len(_AMPRION_VOLTAGE_ORDER):
        raise ValueError(
            f"Expected {len(_AMPRION_VOLTAGE_ORDER)} Amprion voltage-level rows, "
            f"found {len(data_rows)}. PDF layout may have changed - inspect page 1 manually."
        )

    records = []
    for voltage_level, row in zip(_AMPRION_VOLTAGE_ORDER, data_rows):
        low_leistung, low_arbeit, high_leistung, high_arbeit = row
        records.append({
            "voltage_level": voltage_level,
            "low_util_leistungspreis_eur_kwa": _parse_de_number(low_leistung),
            "low_util_arbeitspreis_ct_kwh": _parse_de_number(low_arbeit),
            "high_util_leistungspreis_eur_kwa": _parse_de_number(high_leistung),
            "high_util_arbeitspreis_ct_kwh": _parse_de_number(high_arbeit),
        })
    return pd.DataFrame(records)


def _is_numeric_de(text: str) -> bool:
    try:
        _parse_de_number(text)
        return True
    except (ValueError, AttributeError):
        return False


def normalize_tariffs(westnetz_df: pd.DataFrame, amprion_df: pd.DataFrame) -> pd.DataFrame:
    """Combine both operators' extracted tariffs into one canonical table,
    tagged by operator."""
    westnetz_df = westnetz_df.copy()
    westnetz_df.insert(0, "operator", "westnetz")
    amprion_df = amprion_df.copy()
    amprion_df.insert(0, "operator", "amprion")
    combined = pd.concat([westnetz_df, amprion_df], ignore_index=True)
    combined["voltage_level"] = combined["voltage_level"].astype(str)
    return combined


def get_default_tariff(tariffs_df: pd.DataFrame) -> dict:
    """The single tariff row used for all customers in this project
    model (see DEFAULT_WESTNETZ_VOLTAGE_LEVEL)."""
    match = tariffs_df[
        (tariffs_df["operator"] == "westnetz")
        & (tariffs_df["voltage_level"] == DEFAULT_WESTNETZ_VOLTAGE_LEVEL)
    ]
    if len(match) != 1:
        raise ValueError(f"Expected exactly one default tariff row, found {len(match)}.")
    return match.iloc[0].to_dict()
