"""
diagnosis.py
------------
The app's actual runtime path: ONE customer's raw electricity time series
in, one LLM-ready diagnostic record out.

This is deliberately separate from the training pipeline (notebooks 00,
02-07), which is a bulk, offline, occasional process whose only job is to
produce trained ARTIFACTS - the K-Means pipeline (.joblib) and the
normalized tariff table. The app never touches the 250-customer synthetic
population directly; it only ever loads those pre-trained artifacts and
runs ONE customer through them. Every module this file calls
(peaks/features/financial/llm_profile) was already written generically
enough to take a single customer - this module is the orchestration that
was missing, not new modeling logic.

K-Means is used in PREDICT mode only here - never fit. Retraining the
model is exclusively the training pipeline's job (notebook 03).

Scope: electricity only, matching financial.py and llm_profile.py's
current scope (no gas tariff data exists yet).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import pandas as pd

import clustering
import features
import financial
import llm_profile
import peaks
import pricing

DEFAULT_KMEANS_PATH = "../models/kmeans_pipeline_electricity.joblib"
DEFAULT_TARIFFS_PATH = "../data/outputs/pricing/network_tariffs_normalized.csv"
DEFAULT_REDUCTION_PERCENTAGES = (5, 10, 15, 20, 30)


def diagnose_customer_from_features(
    customer_id: str,
    features_df: pd.DataFrame,
    peak_events: pd.DataFrame,
    kmeans_pipeline_path: str | Path = DEFAULT_KMEANS_PATH,
    tariffs_path: str | Path = DEFAULT_TARIFFS_PATH,
    reduction_percentages=DEFAULT_REDUCTION_PERCENTAGES,
) -> dict:
    """Same end result as diagnose_customer, but starting from an
    already-computed feature matrix and peak-events table instead of raw
    electricity data - for callers that already have these (e.g. a
    notebook that ran the peaks/features step separately and wants to
    avoid recomputing them for the same customer). diagnose_customer()
    itself is a thin wrapper around this function - the two must never
    drift apart in behavior.
    """
    # --- Cluster: PREDICT with the pre-trained pipeline, never fit ---
    kmeans_pipeline = joblib.load(kmeans_pipeline_path)
    feature_cols = clustering.feature_columns(features_df)
    cluster_id = int(clustering.predict_cluster(kmeans_pipeline, features_df, feature_cols)[0])
    clusters_df = pd.DataFrame([{"customer_id": customer_id, "cluster": cluster_id}])

    # --- Financial: current cost + a menu of peak-reduction scenarios ---
    tariffs = pd.read_csv(tariffs_path)
    default_tariff = pricing.get_default_tariff(tariffs)
    financials_df = financial.compute_customer_financials(features_df, default_tariff)

    scenario_frames = [
        financial.compute_scenario_for_all(features_df, default_tariff, reduction_pct=pct)
        for pct in reduction_percentages
    ]
    scenarios_df = pd.concat(scenario_frames, ignore_index=True)

    # --- Assemble the LLM-ready record (same builder the batch notebook uses) ---
    return llm_profile.build_customer_record(
        customer_id, features_df, clusters_df, peak_events,
        financials_df=financials_df, scenarios_df=scenarios_df,
    )


def diagnose_customer(
    electricity_df: pd.DataFrame,
    kmeans_pipeline_path: str | Path = DEFAULT_KMEANS_PATH,
    tariffs_path: str | Path = DEFAULT_TARIFFS_PATH,
    reduction_percentages=DEFAULT_REDUCTION_PERCENTAGES,
) -> dict:
    """Run one customer's electricity time series through the full
    single-customer pipeline and return the LLM-ready customer record
    (not the final diagnosis text - see diagnose_customer_full for that).

    `electricity_df` must have the canonical schema (timestamp_local,
    timestamp_utc, customer_id, power_kw) - the same shape produced by
    the single-customer generator, or any future "customer uploads their
    own CSV" path. This function doesn't care which of those it came from.

    Uses the ALREADY-TRAINED K-Means pipeline (predict only) and the
    ALREADY-EXTRACTED tariff table - both produced once, offline, by the
    training pipeline. Nothing is retrained or re-extracted here.

    Computes peaks and features from the raw data, then delegates to
    diagnose_customer_from_features() for everything after that - if you
    already have precomputed peaks/features for this customer (e.g. from
    notebook 02), call that function directly instead to skip the
    recomputation.
    """
    customer_ids = electricity_df["customer_id"].unique()
    if len(customer_ids) != 1:
        raise ValueError(
            f"diagnose_customer expects exactly one customer's data, found {len(customer_ids)}: "
            f"{list(customer_ids)}. For multiple customers, call this once per customer."
        )
    customer_id = customer_ids[0]

    # --- Peaks + features (same functions the training pipeline uses) ---
    peak_events = peaks.detect_all_peaks(electricity_df, value_col="power_kw")
    features_df = features.build_feature_matrix(
        electricity_df, peak_events, value_col="power_kw", value_type="power",
    )

    return diagnose_customer_from_features(
        customer_id, features_df, peak_events,
        kmeans_pipeline_path=kmeans_pipeline_path, tariffs_path=tariffs_path,
        reduction_percentages=reduction_percentages,
    )


def diagnose_customer_full(
    electricity_df: pd.DataFrame,
    generate_fn: Callable[[dict], dict],
    **kwargs,
) -> dict:
    """Convenience wrapper for the app's actual runtime call: build the
    record AND call the LLM to produce the final diagnosis, in one call.

    `generate_fn` is any callable that takes a customer record dict and
    returns a validated diagnosis dict (e.g. notebook 05's generate_profile,
    or the app's own LLM-calling wrapper) - passed in rather than imported,
    so this module never needs an LLM SDK as a dependency and stays fully
    testable offline.
    """
    record = diagnose_customer(electricity_df, **kwargs)
    return generate_fn(record)


def get_peak_events(electricity_df: pd.DataFrame) -> pd.DataFrame:
    """Full per-event peak table (not just the summary folded into
    diagnose_customer's record) - for interactive display, e.g. the app's
    Consumption page peak-events table.

    Deliberately a separate, small recomputation rather than threading this
    through diagnose_customer's return value: that record is also sent
    verbatim to the LLM (see llm_profile.py), which must stay compact - a
    raw per-event table doesn't belong there. The recomputation cost is
    small (peak detection on one customer's year of data)."""
    return peaks.detect_all_peaks(electricity_df, value_col="power_kw")
