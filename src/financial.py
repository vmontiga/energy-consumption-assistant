from __future__ import annotations

import pandas as pd

UTILIZATION_HOURS_THRESHOLD = 2500.0


def utilization_hours(annual_kwh: float, peak_kw: float) -> float:
    """Jahresbenutzungsdauer: hours per year the customer would need to
    run continuously at peak demand to consume their actual annual
    energy. Determines which tariff bracket applies."""
    if peak_kw <= 0:
        return 0.0
    return annual_kwh / peak_kw


def annual_network_cost(annual_kwh: float, peak_kw: float, tariff: dict) -> dict:
    """Annual network cost (Leistungspreis + Arbeitspreis) for one
    customer, given their annual consumption, peak demand, and a tariff
    row (as returned by pricing.get_default_tariff, or any row of the
    normalized tariff table). Returns a full breakdown, not just the
    total, so the caller can see which bracket applied."""
    util_hours = utilization_hours(annual_kwh, peak_kw)
    is_high_utilization = util_hours >= UTILIZATION_HOURS_THRESHOLD

    if is_high_utilization:
        leistungspreis = tariff["high_util_leistungspreis_eur_kwa"]
        arbeitspreis_ct = tariff["high_util_arbeitspreis_ct_kwh"]
    else:
        leistungspreis = tariff["low_util_leistungspreis_eur_kwa"]
        arbeitspreis_ct = tariff["low_util_arbeitspreis_ct_kwh"]

    demand_charge_eur = leistungspreis * peak_kw
    energy_charge_eur = (arbeitspreis_ct / 100.0) * annual_kwh
    total_eur = demand_charge_eur + energy_charge_eur

    return {
        "utilization_hours": util_hours,
        "utilization_bracket": "high (>=2500 h/a)" if is_high_utilization else "low (<2500 h/a)",
        "leistungspreis_eur_kwa": leistungspreis,
        "arbeitspreis_ct_kwh": arbeitspreis_ct,
        "demand_charge_eur": demand_charge_eur,
        "energy_charge_eur": energy_charge_eur,
        "total_network_cost_eur": total_eur,
    }


def compute_customer_financials(
    features_df: pd.DataFrame, tariff: dict, id_col: str = "customer_id"
) -> pd.DataFrame:
    """Vectorized-by-row version of annual_network_cost for a whole
    electricity feature matrix. Needs 'annual_consumption' (annual kWh)
    and 'max_consumption' (peak kW) columns - both already produced by
    features.basic_features()."""
    rows = []
    for _, row in features_df.iterrows():
        result = annual_network_cost(
            annual_kwh=row["annual_consumption"],
            peak_kw=row["max_consumption"],
            tariff=tariff,
        )
        result[id_col] = row[id_col]
        rows.append(result)
    out = pd.DataFrame(rows)
    return out[[id_col] + [c for c in out.columns if c != id_col]]


def peak_reduction_scenario(
    annual_kwh: float, peak_kw: float, tariff: dict, reduction_pct: float,
) -> dict:
    """'What if this customer reduced their peak demand by X%?' - holds
    annual energy constant (see module docstring) and correctly
    RE-SELECTS the utilization-hours bracket for the new (lower) peak,
    since a large enough reduction can push a customer across the
    2500 h/a threshold into a structurally different tariff bracket."""
    baseline = annual_network_cost(annual_kwh, peak_kw, tariff)
    new_peak_kw = peak_kw * (1 - reduction_pct / 100.0)
    scenario = annual_network_cost(annual_kwh, new_peak_kw, tariff)

    savings_eur = baseline["total_network_cost_eur"] - scenario["total_network_cost_eur"]
    savings_pct = (
        savings_eur / baseline["total_network_cost_eur"] * 100.0
        if baseline["total_network_cost_eur"] else 0.0
    )

    return {
        "reduction_pct": reduction_pct,
        "baseline_peak_kw": peak_kw,
        "new_peak_kw": new_peak_kw,
        "baseline_total_cost_eur": baseline["total_network_cost_eur"],
        "new_total_cost_eur": scenario["total_network_cost_eur"],
        "savings_eur": savings_eur,
        "savings_pct": savings_pct,
        "baseline_bracket": baseline["utilization_bracket"],
        "new_bracket": scenario["utilization_bracket"],
        "bracket_changed": baseline["utilization_bracket"] != scenario["utilization_bracket"],
    }


def compute_scenario_for_all(
    features_df: pd.DataFrame, tariff: dict, reduction_pct: float, id_col: str = "customer_id",
) -> pd.DataFrame:
    """peak_reduction_scenario applied to a whole electricity feature
    matrix at once."""
    rows = []
    for _, row in features_df.iterrows():
        result = peak_reduction_scenario(
            annual_kwh=row["annual_consumption"],
            peak_kw=row["max_consumption"],
            tariff=tariff,
            reduction_pct=reduction_pct,
        )
        result[id_col] = row[id_col]
        rows.append(result)
    out = pd.DataFrame(rows)
    return out[[id_col] + [c for c in out.columns if c != id_col]]
