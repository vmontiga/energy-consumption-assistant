from __future__ import annotations

import json

import pandas as pd

# ============================================================
# OUTPUT SCHEMA
# ============================================================

PROFILE_SCHEMA = {
    "name": "customer_diagnostic",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "profile_summary": {
                "type": "string",
                "description": "2-4 sentence overall summary of this customer's consumption profile.",
            },
            "behavior_summary": {
                "type": "string",
                "description": "Interpretation of the customer's behavioral pattern (timing, variability, load factor).",
            },
            "main_cost_driver": {
                "type": ["string", "null"],
                "description": "The single largest driver of this customer's network cost, ONLY if the data clearly supports identifying one. Null if not clearly supported.",
            },
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete, specific observations - each should reference an actual number from the record.",
            },
            "investigation_pointers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "General, rule-of-thumb patterns worth investigating given the key findings above - framed as general domain knowledge (e.g. 'high peak density can indicate redundant machinery or overlapping process schedules'), NOT as customer-specific facts or costed recommendations. No number, saving estimate, or feasibility claim belongs here.",
            },
            "consultant_focus": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific topics/questions the consultant should raise with the customer.",
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Missing information, uncertainties, or scope limits - e.g. 'no financial data available for this customer'.",
            },
        },
        "required": [
            "customer_id", "profile_summary", "behavior_summary", "main_cost_driver",
            "key_findings", "investigation_pointers", "consultant_focus", "caveats",
        ],
        "additionalProperties": False,
    },
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an energy consulting profile analyst. Your reader is a
professional energy CONSULTANT preparing to advise a business customer - not the
customer themselves. Write for that reader: technical terms (kW, utilization hours,
tariff brackets, load factor) are appropriate; vagueness is not.

Python is the source of truth. You receive a structured record of results already
computed by a deterministic pipeline (behavioral features, peak detection, cluster
assignment, and - where available - financial cost figures and strategy scenarios).
Your job is to interpret and explain these results, not to produce new ones.

CLARITY IS THE PRIMARY GOAL. Every sentence should either state a concrete fact
tied to a number in the record, or draw an explicit, well-reasoned conclusion from
one. Avoid generic filler ("this customer shows interesting patterns"), hedge-words
that say nothing ("may potentially sometimes"), and consulting cliches that aren't
backed by the record ("consider exploring efficiency opportunities").

You MUST:
- Preserve every supplied number EXACTLY as given - never restate, round
  differently, or recompute a value.
- Ground every key finding, cost driver, and consultant-focus point in a specific
  field of the record - a consultant reading this should be able to point to the
  exact number behind each claim.
- Identify a main cost driver ONLY when the record clearly supports one; otherwise
  return null and say why in a caveat.
- Clearly distinguish measured facts (behavioral features, detected peaks) from
  general domain heuristics (investigation_pointers) - never blur the two.
- For investigation_pointers specifically (and ONLY there): you may draw on general
  energy-consulting domain knowledge to suggest patterns worth investigating, tied
  to a pattern observed in the key findings (e.g. a high peak count or peak ratio
  found in the record can motivate "high peak density often points to redundant
  machinery or overlapping process schedules - worth checking"). These are
  deliberately general and NOT customer-specific claims: no number, no savings
  estimate, no feasibility claim, no reference to a specific piece of equipment
  this customer is assumed to have.
- Explicitly surface missing or unavailable information as a caveat rather than
  silently omitting it or working around it.

You MUST NOT:
- Invent, estimate, or guess any consumption value, price, saving, peak value, or
  classification not present in the record.
- Perform any calculation yourself - not even simple arithmetic like unit
  conversions or averages not already in the record.
- Introduce outside facts, general industry assumptions, or typical-customer
  comparisons anywhere EXCEPT investigation_pointers, where general (not
  customer-specific) domain heuristics are the explicit point.
- Present an investigation_pointer as if it were a grounded, customer-specific
  finding, a costed recommendation, or a claim of technical feasibility for this
  customer - it is a general pattern to go check, nothing more.
"""


# ============================================================
# CUSTOMER RECORD ASSEMBLY
# ============================================================

_FEATURE_FIELDS = [
    "annual_consumption", "mean_consumption", "max_consumption", "load_factor",
    "coefficient_of_variation", "night_share", "morning_share", "daytime_share",
    "evening_share", "weekday_share", "weekend_share",
]

_PEAK_FIELDS = [
    "peak_count", "max_peak", "mean_peak", "mean_peak_ratio", "max_peak_ratio",
    "mean_peak_duration_hours", "morning_peak_share", "daytime_peak_share",
    "evening_peak_share", "weekday_peak_share",
]

_FINANCIAL_FIELDS = [
    "utilization_hours", "utilization_bracket", "leistungspreis_eur_kwa",
    "arbeitspreis_ct_kwh", "demand_charge_eur", "energy_charge_eur", "total_network_cost_eur",
]


def _row_to_dict(df: pd.DataFrame, customer_id: str, fields: list[str], id_col: str = "customer_id") -> dict | None:
    match = df[df[id_col] == customer_id]
    if match.empty:
        return None
    row = match.iloc[0]
    return {f: (None if pd.isna(row[f]) else round(float(row[f]), 4) if isinstance(row[f], float) else row[f]) for f in fields}


def build_customer_record(
    customer_id: str,
    features_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    peak_events_df: pd.DataFrame,
    financials_df: pd.DataFrame | None = None,
    scenarios_df: pd.DataFrame | None = None,
    top_n_peak_events: int = 3,
) -> dict:
    """Assemble the compact record sent to the LLM for one customer. Never
    includes the raw time series - keeps the prompt small and auditable.
    Missing sections (e.g. financial data, if unavailable for a customer)
    are `None`, never guessed or silently dropped."""
    behavioral = _row_to_dict(features_df, customer_id, _FEATURE_FIELDS)
    if behavioral is None:
        raise ValueError(f"No feature row found for customer_id={customer_id!r}")

    peak_summary = _row_to_dict(features_df, customer_id, _PEAK_FIELDS)

    cluster_match = clusters_df[clusters_df["customer_id"] == customer_id]
    cluster_id = int(cluster_match.iloc[0]["cluster"]) if not cluster_match.empty else None

    events = peak_events_df[peak_events_df["customer_id"] == customer_id].sort_values("peak_value", ascending=False)
    top_events = [
        {
            "peak_time": str(row["peak_time"]),
            "peak_value_kw": round(float(row["peak_value"]), 2),
            "is_weekend": bool(row["is_weekend"]),
        }
        for _, row in events.head(top_n_peak_events).iterrows()
    ]

    financial = None
    if financials_df is not None:
        financial = _row_to_dict(financials_df, customer_id, _FINANCIAL_FIELDS)

    strategies = None
    if scenarios_df is not None:
        match = scenarios_df[scenarios_df["customer_id"] == customer_id]
        if not match.empty:
            strategies = []
            for _, row in match.iterrows():
                strategies.append({
                    "reduction_pct": float(row["reduction_pct"]),
                    "baseline_peak_kw": round(float(row["baseline_peak_kw"]), 2),
                    "new_peak_kw": round(float(row["new_peak_kw"]), 2),
                    "savings_eur": round(float(row["savings_eur"]), 2),
                    "savings_pct": round(float(row["savings_pct"]), 2),
                    "bracket_changed": bool(row["bracket_changed"]),
                    "baseline_bracket": row["baseline_bracket"],
                    "new_bracket": row["new_bracket"],
                })

    return {
        "customer_id": customer_id,
        "cluster_id": cluster_id,
        "behavioral_features": behavioral,
        "peak_analysis": {**(peak_summary or {}), "top_peak_events": top_events},
        "financial": financial,
        "strategies": strategies,
    }


def build_user_message(customer_record: dict) -> str:
    """The user-turn content sent alongside SYSTEM_PROMPT: the record,
    verbatim, as JSON - nothing added, nothing paraphrased."""
    return (
        "Here is the structured record for one customer. Produce the diagnostic "
        "following the required schema, using only the information below:\n\n"
        + json.dumps(customer_record, indent=2, default=str)
    )


def validate_profile(profile: dict) -> list[str]:
    """Lightweight structural check against PROFILE_SCHEMA's required
    fields/types - a safety net regardless of whether the API's structured
    output mode was actually used. Returns a list of problems (empty list
    = valid)."""
    problems = []
    for field in PROFILE_SCHEMA["schema"]["required"]:
        if field not in profile:
            problems.append(f"Missing required field: {field}")

    return problems


def generate_profile(customer_record: dict, model: str = "gpt-4o-2024-08-06") -> dict:
    """Call the LLM to produce the diagnosis for one customer record.
    Import of the openai SDK is local to this function (not at module
    level) so llm_profile.py never requires openai to be installed just
    to build records/validate schemas - only this one function needs it,
    and only when actually called."""
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from the environment automatically

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(customer_record)},
        ],
        response_format={"type": "json_schema", "json_schema": PROFILE_SCHEMA},
    )
    profile = json.loads(response.choices[0].message.content)
    problems = validate_profile(profile)
    if problems:
        raise ValueError(f"Generated profile failed validation: {problems}")
    return profile
