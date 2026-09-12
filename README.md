# Energy Consumption Diagnostic

## Structure

```
notebooks/    00, 01, 02, 03 - the training pipeline (bulk, offline, run occasionally).
              Produces the trained artifacts below. Never touched by the app itself.
              04, 05 - single-customer demonstration notebooks (financial calc, LLM
              diagnostic), each generating one fresh test customer. Not required for
              the app to run - useful for reviewing/testing that logic in isolation.
models/       Trained K-Means pipelines (.joblib) - produced by notebook 03.
data/         inputs/raw (official tariff PDFs), inputs/generated (synthetic
              population, for training only), outputs (all pipeline results).
src/          Shared modules imported by both the notebooks and app.py.
app.py        The Streamlit app - upload a consumption file, get a diagnostic, 3 tabs.
```

## Setup

```bash
pip install -r requirements.txt
```

Create a file named `.env` next to `app.py`, containing your OpenAI key (used only for the
"Generate diagnosis & strategy" button - everything else in the app works without it):

```
OPENAI_API_KEY=sk-...
```

## One-time: run the training pipeline

Run notebooks in order once (and again any time the underlying data or logic changes):

```
00_synthetic_customer_generator.ipynb
01_pricing_collection_and_normalization.ipynb
02_peaks_and_features.ipynb
03_kmeans_clustering.ipynb
```

This produces `models/kmeans_pipeline_electricity.joblib` and
`data/outputs/pricing/network_tariffs_normalized.csv`, which `app.py` loads directly.

`04_financial_analysis.ipynb` and `05_llm_profile_diagnostic.ipynb` are optional demos of the
financial/diagnostic logic, but **not independent** of the notebooks above - both load the single
test customer that `00_synthetic_customer_generator.ipynb` generates and
`02_peaks_and_features.ipynb` computes peaks/features for (via a `LATEST_CUSTOMER_ID.txt` pointer
file in `data/inputs/generated/single_test/`), so run those two first. This keeps 04 and 05 looking
at the *same* customer instead of each generating their own, and avoids recomputing peaks/features
notebook 02 already produced. Re-run 00 (then 02) for a different customer.

## Run the app

```bash
streamlit run app.py
```

Upload a consumption file in the sidebar to begin - CSV or Excel, with at least
`timestamp_utc`, `customer_id`, and `power_kw` columns (see `src/uploads.py`). No test data on
hand? The sidebar's **Need a test file?** section generates one fully-randomized customer
(known archetype, useful for sanity-checking the predicted cluster) as a downloadable file in
the exact format the app expects - upload it right back in to try the app end to end.

## Scope (current)

- Electricity only - no gas tariff data has been collected yet.
- Financial figures use a single fixed network tariff profile (see `src/pricing.py`).
- K-Means is used in *predict* mode only in the app - retraining only happens in notebook 03.
- Upload parsing (`src/uploads.py`) expects the canonical single-customer schema already -
  it isn't a general raw-file parser (no German number formats, no multi-customer sheets).



