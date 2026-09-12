from __future__ import annotations

import numpy as np
import pandas as pd

import pricing

TIMEZONE = "Europe/Berlin"

# ============================================================
# ELECTRICITY ARCHETYPES
# ============================================================
# annual_kwh: target annual electricity consumption range
# base_load_ratio / daytime / night multipliers: shape of the demand curve
# peak_probability / peak_multiplier: recurring random demand spikes
# operating_hours: informational only (not currently used in generation)

ELECTRICITY_ARCHETYPES = {
    "small_commercial": {
        "annual_kwh": (250_000, 800_000),
        "base_load_ratio": (0.18, 0.30),
        "daytime_multiplier": (1.7, 2.4),
        "night_multiplier": (0.75, 0.95),
        "peak_probability": (0.015, 0.030),
        "peak_multiplier": (1.25, 1.60),
        "operating_hours": (8, 11),
    },
    "medium_commercial": {
        "annual_kwh": (800_000, 2_500_000),
        "base_load_ratio": (0.25, 0.38),
        "daytime_multiplier": (1.6, 2.2),
        "night_multiplier": (0.80, 0.98),
        "peak_probability": (0.012, 0.025),
        "peak_multiplier": (1.20, 1.55),
        "operating_hours": (9, 13),
    },
    "industrial": {
        "annual_kwh": (2_000_000, 8_000_000),
        "base_load_ratio": (0.45, 0.65),
        "daytime_multiplier": (1.20, 1.55),
        "night_multiplier": (0.90, 1.05),
        "peak_probability": (0.008, 0.018),
        "peak_multiplier": (1.15, 1.40),
        "operating_hours": (12, 20),
    },
    "intensive_industrial": {
        "annual_kwh": (6_000_000, 25_000_000),
        "base_load_ratio": (0.60, 0.78),
        "daytime_multiplier": (1.10, 1.35),
        "night_multiplier": (0.95, 1.08),
        "peak_probability": (0.005, 0.012),
        "peak_multiplier": (1.10, 1.30),
        "operating_hours": (18, 24),
    },
    "flexible_peaky": {
        "annual_kwh": (1_000_000, 6_000_000),
        "base_load_ratio": (0.25, 0.45),
        "daytime_multiplier": (1.7, 2.5),
        "night_multiplier": (0.70, 0.95),
        "peak_probability": (0.025, 0.055),
        "peak_multiplier": (1.35, 1.90),
        "operating_hours": (8, 14),
    },
}

# Sampling weights for the bulk population (must align with archetype order)
ELECTRICITY_ARCHETYPE_WEIGHTS = [0.20, 0.28, 0.27, 0.10, 0.15]

# Archetypes where weekends are noticeably quieter (commercial-style)
WEEKEND_QUIET_ARCHETYPES = {"small_commercial", "medium_commercial", "flexible_peaky"}


# ============================================================
# GAS ARCHETYPES
# ============================================================
# annual_kwh_ratio_to_electricity: gas annual energy is scaled off the
# company's electricity annual consumption, so gas magnitude stays coherent
# with company size without needing its own independent absolute range.

GAS_ARCHETYPES = {
    "heating_gas": {
        # Strongly seasonal: near-zero in summer, peaks in winter.
        "annual_kwh_ratio_to_electricity": (0.5, 1.5),
        "summer_floor_ratio": (0.02, 0.08),
        "daytime_multiplier": (1.1, 1.4),
        "night_multiplier": (0.85, 0.98),
    },
    "process_gas": {
        # Roughly flat year-round, driven by operating schedule not weather.
        "annual_kwh_ratio_to_electricity": (0.8, 2.0),
        "base_load_ratio": (0.55, 0.75),
        "daytime_multiplier": (1.05, 1.25),
        "night_multiplier": (0.90, 1.05),
        "peak_probability": (0.010, 0.020),
        "peak_multiplier": (1.15, 1.35),
    },
}

# Which gas archetype (if any) fits each electricity archetype, and how
# likely that company is to have a gas connection at all. This keeps a given
# company's electricity and gas behavior coherent instead of independently
# random.
ARCHETYPE_ENERGY_CONFIG = {
    "small_commercial": {"gas_probability": 0.60, "gas_archetype": "heating_gas"},
    "medium_commercial": {"gas_probability": 0.60, "gas_archetype": "heating_gas"},
    "industrial": {"gas_probability": 0.70, "gas_archetype": "process_gas"},
    "intensive_industrial": {"gas_probability": 0.80, "gas_archetype": "process_gas"},
    "flexible_peaky": {"gas_probability": 0.30, "gas_archetype": "heating_gas"},
}


# ============================================================
# GRID CONNECTION (electricity metadata only - used later for tariffs)
# ============================================================

def choose_grid_connection(annual_kwh: float, rng: np.random.Generator):
    """Simplification, per project decision: every synthetic customer
    uses the SAME single tariff profile (pricing.DEFAULT_WESTNETZ_VOLTAGE_LEVEL),
    rather than a plausible-but-unvalidated mix of voltage levels. `annual_kwh`
    and `rng` are kept as parameters (unused) so this function's call sites
    don't need to change if per-customer voltage-level variation is
    reintroduced later - see pricing.py for the full extracted tariff table,
    which already supports selecting a different row per customer whenever
    that's wanted."""
    voltage = pricing.DEFAULT_WESTNETZ_VOLTAGE_LEVEL
    metering = "registered_load_profile"
    return voltage, metering


# ============================================================
# TIME INDEX HELPERS
# ============================================================

def _year_index(year: int, freq: str) -> pd.DatetimeIndex:
    """Timezone-aware Europe/Berlin index for one full calendar year, at the
    given resolution. pandas handles the DST fall-back/spring-forward
    transitions correctly by construction (see module docstring)."""
    return pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year}-12-31 23:59",
        freq=freq,
        tz=TIMEZONE,
    )


def _interval_hours(freq: str) -> float:
    """Length, in hours, of one step at the given pandas frequency string."""
    from pandas.tseries.frequencies import to_offset
    return pd.Timedelta(to_offset(freq)).total_seconds() / 3600


# ============================================================
# ELECTRICITY PROFILE GENERATION (15-minute resolution)
# ============================================================

def make_electricity_profile(
    archetype: str,
    annual_kwh: float,
    year: int,
    rng: np.random.Generator,
    freq: str = "15min",
) -> pd.DataFrame:
    """Generate one calendar year of electricity demand (power_kw) at the
    given resolution (default 15-minute)."""
    cfg = ELECTRICITY_ARCHETYPES[archetype]

    idx = _year_index(year, freq)
    df = pd.DataFrame({"timestamp_local": idx})
    hour = df["timestamp_local"].dt.hour.to_numpy()
    weekday = df["timestamp_local"].dt.weekday.to_numpy()
    month = df["timestamp_local"].dt.month.to_numpy()

    seasonal = 1.0 + 0.10 * np.cos((month - 1) / 12 * 2 * np.pi)

    is_weekday = weekday < 5
    daytime = (hour >= 7) & (hour <= 19)

    base_ratio = rng.uniform(*cfg["base_load_ratio"])
    day_mult = rng.uniform(*cfg["daytime_multiplier"])
    night_mult = rng.uniform(*cfg["night_multiplier"])
    peak_prob = rng.uniform(*cfg["peak_probability"])
    peak_mult = rng.uniform(*cfg["peak_multiplier"])

    profile = np.ones(len(df)) * base_ratio
    profile[daytime] *= day_mult
    profile[~daytime] *= night_mult

    weekend_factor = 0.72 if archetype in WEEKEND_QUIET_ARCHETYPES else 0.92
    profile[~is_weekday] *= weekend_factor

    noise = rng.normal(1.0, 0.055, len(df))
    noise = np.clip(noise, 0.82, 1.20)
    profile *= seasonal * noise

    peak_mask = rng.random(len(df)) < peak_prob
    profile[peak_mask] *= rng.uniform(peak_mult * 0.85, peak_mult * 1.15, peak_mask.sum())

    profile = np.clip(profile, 0.05, None)

    # Scale so that sum(power_kw * interval_hours) matches the annual kWh target.
    interval_hours = _interval_hours(freq)
    profile *= annual_kwh / (profile.sum() * interval_hours)

    df["power_kw"] = profile
    df["timestamp_utc"] = df["timestamp_local"].dt.tz_convert("UTC")
    return df


# ============================================================
# GAS PROFILE GENERATION (hourly resolution)
# ============================================================

def make_gas_profile(
    archetype: str,
    annual_kwh: float,
    year: int,
    rng: np.random.Generator,
    freq: str = "h",
) -> pd.DataFrame:
    """Generate one calendar year of gas energy consumption (energy_kwh) at
    the given resolution (default hourly)."""
    cfg = GAS_ARCHETYPES[archetype]

    idx = _year_index(year, freq)
    df = pd.DataFrame({"timestamp_local": idx})
    hour = df["timestamp_local"].dt.hour.to_numpy()
    weekday = df["timestamp_local"].dt.weekday.to_numpy()
    month = df["timestamp_local"].dt.month.to_numpy()
    is_weekday = weekday < 5
    daytime = (hour >= 7) & (hour <= 19)

    day_mult = rng.uniform(*cfg["daytime_multiplier"])
    night_mult = rng.uniform(*cfg["night_multiplier"])

    if archetype == "heating_gas":
        summer_floor = rng.uniform(*cfg["summer_floor_ratio"])
        # 1.0 in mid-winter (Jan), ~0 in mid-summer (Jul), floored at summer_floor.
        raw = (1 + np.cos((month - 1) / 12 * 2 * np.pi)) / 2  # 1 in Jan, 0 in Jul
        seasonal = summer_floor + (1 - summer_floor) * raw ** 1.5

        profile = seasonal.astype(float)
        profile[daytime] *= day_mult
        profile[~daytime] *= night_mult

        noise = rng.normal(1.0, 0.06, len(df))
        noise = np.clip(noise, 0.80, 1.20)
        profile *= noise

    elif archetype == "process_gas":
        base_ratio = rng.uniform(*cfg["base_load_ratio"])
        peak_prob = rng.uniform(*cfg["peak_probability"])
        peak_mult = rng.uniform(*cfg["peak_multiplier"])

        profile = np.ones(len(df)) * base_ratio
        profile[daytime] *= day_mult
        profile[~daytime] *= night_mult
        # Process gas is much less weekend-sensitive than commercial electricity,
        # but still somewhat lower on weekends (reduced shifts).
        profile[~is_weekday] *= 0.85

        noise = rng.normal(1.0, 0.05, len(df))
        noise = np.clip(noise, 0.82, 1.18)
        profile *= noise

        peak_mask = rng.random(len(df)) < peak_prob
        profile[peak_mask] *= rng.uniform(peak_mult * 0.85, peak_mult * 1.15, peak_mask.sum())

    else:
        raise ValueError(f"Unknown gas archetype: {archetype}")

    profile = np.clip(profile, 0.0, None)

    interval_hours = _interval_hours(freq)
    current_total = profile.sum() * interval_hours
    if current_total > 0:
        profile *= annual_kwh / current_total

    df["energy_kwh"] = profile
    df["timestamp_utc"] = df["timestamp_local"].dt.tz_convert("UTC")
    return df


# ============================================================
# SINGLE-COMPANY GENERATION (shared by bulk loop and single-customer notebook)
# ============================================================

def generate_company(
    customer_id: str,
    rng: np.random.Generator,
    year: int = 2026,
    electricity_archetype: str | None = None,
    force_gas: bool | None = None,
) -> dict:
    """Generate one company's full data: metadata + electricity profile +
    (possibly) gas profile, following the same rules used by the bulk
    population generator.

    electricity_archetype: pass None to randomize (weighted like the bulk
        population); pass an explicit archetype name to control it.
    force_gas: pass None to let gas presence be randomized per
        ARCHETYPE_ENERGY_CONFIG's probability; pass True/False to force
        presence/absence (e.g. for testing a specific scenario).

    Returns a dict with keys: metadata (dict), electricity (DataFrame or
    None), gas (DataFrame or None).
    """
    archetypes = list(ELECTRICITY_ARCHETYPES)

    if electricity_archetype is None:
        electricity_archetype = rng.choice(archetypes, p=ELECTRICITY_ARCHETYPE_WEIGHTS)
    elif electricity_archetype not in ELECTRICITY_ARCHETYPES:
        raise ValueError(f"Unknown electricity archetype: {electricity_archetype}")

    elec_cfg = ELECTRICITY_ARCHETYPES[electricity_archetype]
    annual_electricity_kwh = rng.uniform(*elec_cfg["annual_kwh"])

    voltage, metering = choose_grid_connection(annual_electricity_kwh, rng)

    electricity_df = make_electricity_profile(
        electricity_archetype, annual_electricity_kwh, year, rng
    )
    electricity_df["customer_id"] = customer_id
    electricity_df["source_file"] = "synthetic_generator"

    energy_cfg = ARCHETYPE_ENERGY_CONFIG[electricity_archetype]
    has_gas = (
        force_gas if force_gas is not None
        else bool(rng.random() < energy_cfg["gas_probability"])
    )

    gas_archetype = None
    annual_gas_kwh = None
    gas_df = None

    if has_gas:
        gas_archetype = energy_cfg["gas_archetype"]
        gas_cfg = GAS_ARCHETYPES[gas_archetype]
        ratio = rng.uniform(*gas_cfg["annual_kwh_ratio_to_electricity"])
        annual_gas_kwh = annual_electricity_kwh * ratio

        gas_df = make_gas_profile(gas_archetype, annual_gas_kwh, year, rng)
        gas_df["customer_id"] = customer_id
        gas_df["source_file"] = "synthetic_generator"

    metadata = {
        "customer_id": customer_id,
        "customer_type": "company",
        "pricing_year": year,
        "electricity_archetype": electricity_archetype,
        "gas_archetype": gas_archetype,
        "grid_voltage_level": voltage,
        "metering_type": metering,
        "has_electricity": True,
        "has_gas": has_gas,
        "target_annual_electricity_kwh": annual_electricity_kwh,
        "target_annual_gas_kwh": annual_gas_kwh,
    }

    return {
        "metadata": metadata,
        "electricity": electricity_df,
        "gas": gas_df,
    }


# ============================================================
# BULK POPULATION GENERATION
# ============================================================

def generate_company_population(
    n_companies: int = 250,
    year: int = 2026,
    seed: int = 42,
    id_prefix: str = "NRW_COMP",
):
    """Generate a full synthetic population. Returns:
    metadata_df, electricity_long_df, gas_long_df

    electricity_long_df columns: timestamp_local | timestamp_utc | customer_id | power_kw | source_file
    gas_long_df columns:         timestamp_local | timestamp_utc | customer_id | energy_kwh | source_file
    (gas_long_df only contains rows for customers with has_gas=True)
    """
    rng = np.random.default_rng(seed)

    metadata_rows = []
    electricity_frames = []
    gas_frames = []

    for i in range(1, n_companies + 1):
        customer_id = f"{id_prefix}_{i:04d}"
        result = generate_company(customer_id, rng, year=year)

        metadata_rows.append(result["metadata"])
        electricity_frames.append(result["electricity"])
        if result["gas"] is not None:
            gas_frames.append(result["gas"])

    metadata_df = pd.DataFrame(metadata_rows)
    electricity_long_df = pd.concat(electricity_frames, ignore_index=True)
    gas_long_df = (
        pd.concat(gas_frames, ignore_index=True)
        if gas_frames
        else pd.DataFrame(columns=["timestamp_local", "timestamp_utc", "energy_kwh", "customer_id", "source_file"])
    )

    return metadata_df, electricity_long_df, gas_long_df


# ============================================================
# SINGLE RANDOM TEST CUSTOMER
# ============================================================

def generate_single_test_customer(
    year: int = 2026,
    seed: int | None = None,
    id_prefix: str = "NRW_TEST",
):
    """Generate exactly one fully-randomized company, following the same
    rules as the bulk population (same archetype pool, same weights, same
    electricity<->gas coherence rule). Uses its own id_prefix/namespace so
    test customers never collide with the training population.

    Returns the same dict shape as generate_company():
    {"metadata": {...}, "electricity": df, "gas": df or None}
    """
    rng = np.random.default_rng(seed)
    suffix = rng.integers(0, 1_000_000)
    customer_id = f"{id_prefix}_{suffix:06d}"

    return generate_company(customer_id, rng, year=year)
