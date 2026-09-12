"""
app.py
------
Streamlit app: single-customer energy diagnostic.

Primary workflow: the user uploads their own consumption file (CSV or Excel,
in the canonical schema - see src/uploads.py), which is run through the full
diagnosis pipeline (predict-only, using pre-trained artifacts from the
training pipeline - notebooks 00-03), and presented across Dashboard /
Consumption / Diagnostic & Strategy views. This is a direct wrapper around
src/diagnosis.py - the Streamlit layer adds presentation only, no new logic.

Test files can be produced with the single-customer generator
(src/synthetic_generator.py, or notebook 00's "Single Customer Generation
for Testing Purposes" section) - saved in the exact format this app's
uploader expects.

The bulk 250-customer training population is never loaded here - the app
only ever handles ONE customer at a time, matching how it will actually be
used (a consultant diagnosing one customer's consumption before a call).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

import diagnosis as dx  # noqa: E402
import llm_profile as lp  # noqa: E402
import plots as pl  # noqa: E402
import uploads  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # override=True: a fresh .env value always wins over
                                 # anything already sitting in the environment (a stale
                                 # shell var, an old placeholder) - load_dotenv()'s
                                 # default silently keeps the OLD value otherwise
except Exception:
    pass  # fine either way - falls back to any OPENAI_API_KEY already set
          # in the shell/deployment environment

MODELS_DIR = Path(__file__).parent / "models"
OUTPUTS_DIR = Path(__file__).parent / "data" / "outputs"
KMEANS_PATH = MODELS_DIR / "kmeans_pipeline_electricity.joblib"
TARIFFS_PATH = OUTPUTS_DIR / "pricing" / "network_tariffs_normalized.csv"

st.set_page_config(page_title="Energy Consumption Diagnostic", page_icon="⚡", layout="wide")

# EASI color map applied via CSS injection, since Streamlit's [theme] config
# (.streamlit/config.toml) only covers a few broad slots (background, sidebar,
# primary accent, text) - card backgrounds, borders, and secondary text need
# CSS targeting Streamlit's internal component structure directly.
#
# Confidence note: the data-testid attributes used below (stMetric,
# stVerticalBlockBorderWrapper, stCaptionContainer, stBaseButton-secondary)
# and the st-key-<name> class from st.container(key=...) are Streamlit's
# actual internal hooks as of recent versions, but Streamlit doesn't
# document them as a stable public API - they can change between releases.
# If any of this doesn't visibly apply once you run the app, that's the
# first thing to check (browser devtools -> inspect the element -> find its
# current data-testid/class) and I can adjust the selector.
st.markdown("""
<style>
/* Metric cards: title stays at its natural top position; the value is
   vertically centered in whatever space remains below it (not just
   horizontally centered where it naturally falls right under the title).
   Only has a visible effect on cards with an explicit container height
   (see the height=150/300 containers throughout the pages) - on a card
   with no fixed height there's no extra vertical space to center within,
   so this degrades harmlessly there. */
[data-testid="stMetric"] {
    text-align: center;
    height: 100%;
    display: flex;
    flex-direction: column;
}
[data-testid="stMetricLabel"] { justify-content: center; }
[data-testid="stMetricLabel"] > div { justify-content: center; }
[data-testid="stMetricValue"] {
    color: #FFFFFF;
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
}
[data-testid="stMetricLabel"] { color: #A7B5A8; }

/* Custom icon-metric cards (built via markdown, not st.metric() - Streamlit's
   native metric widget has no slot for a custom icon). Mirrors the same
   layout as the stMetric rules above: label stays at its natural top
   position, the icon+value row is vertically centered in whatever space
   remains below it. */
.icon-metric { display: flex; flex-direction: column; height: 100%; text-align: center; }
.icon-metric-label { font-size: 0.85rem; color: #A7B5A8; margin-bottom: 0.15rem; }
.icon-metric-value-row { flex: 1; display: flex; align-items: center; justify-content: center; gap: 0.7rem; }
.icon-metric-badge {
    width: 46px; height: 46px; border-radius: 50%;
    border: 1.5px solid #22C55E; background: rgba(34, 197, 94, 0.1);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.icon-metric-value { font-size: 1.35rem; font-weight: 600; color: #FFFFFF; white-space: nowrap; }

/* Bordered containers (st.container(border=True)) -> Card Background + Border/Divider */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #12332B;
    border-color: #1E4D42 !important;
}

/* Captions (st.caption()) -> Text Secondary */
[data-testid="stCaptionContainer"] { color: #A7B5A8; }

/* Sidebar page nav (scoped to the st.container(key="nav_menu") wrapping the
   3 page buttons only - so this doesn't also restyle the "Generate
   diagnosis & strategy" button elsewhere in the sidebar): big, left-aligned,
   no border on the inactive state - just plain text until active/hovered. */
.st-key-nav_menu .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.85rem 1rem !important;
    font-size: 1rem !important;
    border: none !important;
}
.st-key-nav_menu [data-testid="stBaseButton-secondary"] { background-color: transparent !important; }
.st-key-nav_menu [data-testid="stBaseButton-secondary"]:hover { background-color: #12332B !important; color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

for key in ["electricity_df", "uploaded_file_identity", "record", "diagnosis", "diagnosis_error"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ============================================================
# FORMATTING HELPERS
# ============================================================

def fmt_eur(value) -> str:
    return "—" if value is None or pd.isna(value) else f"€{value:,.0f}"


def fmt_kw(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.1f} kW"


def fmt_annual_consumption(value) -> str:
    """Auto-scales to kWh / MWh / GWh depending on magnitude, rather than
    always showing raw kWh - keeps large customers' numbers readable."""
    if value is None or pd.isna(value):
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} GWh"
    if value >= 1_000:
        return f"{value / 1_000:.0f} MWh"
    return f"{value:.0f} kWh"


# ============================================================
# ICON-METRIC CARDS
# ============================================================
# Reproduces the reference dashboard's per-card icons. Streamlit's native
# st.metric() has no icon slot, so these cards are built as custom HTML via
# st.markdown() instead - see the .icon-metric CSS rules above. Icon SVGs
# and sizes were tuned and visually verified against a rendered mockup
# (not just written and assumed correct) - the different pixel sizes per
# icon (24/30/24/26/24) are deliberate: a thin-stroke icon like the pulse
# line reads as visually smaller than a solid shape like the bolt at the
# same nominal size, so line icons are sized up slightly to look the same
# weight as the filled ones, not literally the same pixel dimensions.

ICON_BARS = '<svg viewBox="0 0 24 24" width="24" height="24"><rect x="1" y="16" width="4" height="7" rx="1" fill="#22C55E"/><rect x="7" y="11" width="4" height="12" rx="1" fill="#22C55E"/><rect x="13" y="6" width="4" height="17" rx="1" fill="#22C55E"/><rect x="19" y="1" width="4" height="22" rx="1" fill="#22C55E"/></svg>'
ICON_PULSE = '<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="#22C55E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1,14 6,14 9,5 13,21 16,9 19,14 23,14"/></svg>'
ICON_BOLT = '<svg viewBox="0 0 24 24" width="24" height="24"><polygon points="13,1 4,14 11,14 9,23 20,9 13,9" fill="#22C55E"/></svg>'
ICON_GAUGE = '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="#22C55E" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="19" x2="19" y2="5"/><polyline points="9,5 19,5 19,15"/></svg>'
ICON_MONEY = '<svg viewBox="0 0 24 24" width="24" height="24"><text x="12" y="17.5" font-size="22" font-weight="700" fill="#22C55E" text-anchor="middle" font-family="Arial, sans-serif">&#8364;</text></svg>'


def render_icon_metric(label: str, value: str, icon: str) -> None:
    """Render a metric card with an icon badge, via markdown - the .icon-metric
    CSS (injected above) handles the label-at-top / value-centered-below
    layout, matching the same visual pattern as the native st.metric() cards
    elsewhere in the app. Caller is responsible for wrapping this in
    st.container(border=True, height=...) - this only renders the inner
    content, not the card's own border/background (the container already
    provides that, matching every other card in the app)."""
    st.markdown(f"""
<div class="icon-metric">
    <div class="icon-metric-label">{label}</div>
    <div class="icon-metric-value-row">
        <span class="icon-metric-badge">{icon}</span>
        <span class="icon-metric-value">{value}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ Energy Diagnostic")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload your consumption file", type=["csv", "xlsx", "xls"], label_visibility="collapsed",
)

if uploaded_file is None:
    st.title("⚡ Energy Consumption Diagnostic")
    st.info("👈 Upload your consumption file in the sidebar to begin.")
    st.stop()

file_identity = (uploaded_file.name, uploaded_file.size)
if st.session_state.uploaded_file_identity != file_identity:
    try:
        st.session_state.electricity_df = uploads.parse_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        st.session_state.uploaded_file_identity = file_identity
        st.session_state.record = None
        st.session_state.diagnosis = None
        st.session_state.diagnosis_error = None
    except uploads.UploadError as exc:
        st.sidebar.error(str(exc))
        st.stop()

electricity_df = st.session_state.electricity_df
customer_id = electricity_df["customer_id"].iloc[0]

st.sidebar.success(f"**{customer_id}**")

if st.session_state.record is None:
    with st.spinner("Analyzing consumption..."):
        try:
            st.session_state.record = dx.diagnose_customer(
                electricity_df, kmeans_pipeline_path=KMEANS_PATH, tariffs_path=TARIFFS_PATH,
            )
        except FileNotFoundError as exc:
            st.error(
                f"Missing trained artifact: {exc}\n\n"
                "Run the training pipeline notebooks (00-03) at least once before using this app."
            )
            st.stop()

record = st.session_state.record

st.sidebar.divider()
st.sidebar.subheader("AI diagnosis")
if st.sidebar.button("🤖 Generate diagnosis & strategy", use_container_width=True):
    with st.spinner("Calling the LLM..."):
        try:
            st.session_state.diagnosis = lp.generate_profile(record)
            st.session_state.diagnosis_error = None
        except Exception as exc:  # noqa: BLE001 - surface any failure (auth, network, schema) to the user
            st.session_state.diagnosis_error = str(exc)

if st.session_state.diagnosis_error:
    st.sidebar.error(f"Diagnosis failed: {st.session_state.diagnosis_error}")
elif st.session_state.diagnosis:
    st.sidebar.success("Diagnosis ready.")
else:
    st.sidebar.caption("Not generated yet. Requires OPENAI_API_KEY (see README / .env).")

st.sidebar.divider()
PAGES = ["Dashboard", "Consumption", "Diagnostic & Strategy"]


def _set_page(label: str) -> None:
    # Using on_click (not `if st.button(...): st.session_state.page = label`)
    # matters here: on_click runs BEFORE the script re-renders, so the
    # just-clicked button already sees the new session_state.page value on
    # its very next draw and shows as selected immediately. The if-pattern
    # updates session_state AFTER that button has already been drawn for
    # this pass, so the selected/green state lags one click behind.
    st.session_state.page = label


with st.sidebar.container(key="nav_menu"):
    for label in PAGES:
        is_active = st.session_state.page == label
        st.button(
            label, type="primary" if is_active else "secondary", use_container_width=True,
            on_click=_set_page, args=(label,),
        )
page = st.session_state.page


# ============================================================
# SHARED RENDER HELPER (used by both Dashboard and the Diagnostic tab)
# ============================================================

def render_short_diagnosis(diagnosis: dict | None) -> None:
    if diagnosis is None:
        st.info("No AI diagnosis generated yet. Click **Generate diagnosis & strategy** in the sidebar.")
        return

    st.markdown(diagnosis["profile_summary"])
    if diagnosis.get("main_cost_driver"):
        st.markdown(f"**Main cost driver:** {diagnosis['main_cost_driver']}")

    top_findings = diagnosis.get("key_findings", [])[:3]
    if top_findings:
        st.markdown("**Top findings:**")
        for finding in top_findings:
            st.markdown(f"- {finding}")

    top_pointers = diagnosis.get("investigation_pointers", [])[:1]
    if top_pointers:
        st.markdown(f"**Worth investigating:** {top_pointers[0]}")

    st.caption("See **Diagnostic & Strategy** in the sidebar for the full analysis.")


# ============================================================
# PAGES
# ============================================================

# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------
if page == "Dashboard":
    st.title("Dashboard")
    st.caption(f"Customer: **{record['customer_id']}**  ·  Predicted cluster: **{record['cluster_id']}**")

    behavioral = record["behavioral_features"]
    peak = record["peak_analysis"]
    financial = record["financial"]
    feat = pl.add_time_features(electricity_df)

    # --- Row 1: key metrics (each in its own bordered card, for equal visual weight) ---
    cols = st.columns(4)
    with cols[0]:
        with st.container(border=True, height=150):
            render_icon_metric("Annual Consumption", fmt_annual_consumption(behavioral["annual_consumption"]), ICON_BARS)
    with cols[1]:
        with st.container(border=True, height=150):
            render_icon_metric("Average Load", fmt_kw(behavioral["mean_consumption"]), ICON_PULSE)
    with cols[2]:
        with st.container(border=True, height=150):
            render_icon_metric("Peak Load", fmt_kw(behavioral["max_consumption"]), ICON_BOLT)
            top_peaks = peak.get("top_peak_events") or []
            if top_peaks:
                peak_time = pd.to_datetime(top_peaks[0]["peak_time"]).strftime("%d.%m.%Y %H:%M")
                st.caption(f"Recorded on {peak_time}")
    with cols[3]:
        with st.container(border=True, height=150):
            mean_consumption = behavioral.get("mean_consumption")
            peak_avg_ratio = behavioral["max_consumption"] / mean_consumption if mean_consumption else None
            render_icon_metric("Peak / Average Ratio", f"{peak_avg_ratio:.2f}" if peak_avg_ratio is not None else "—", ICON_GAUGE)

    # --- Row 2: yearly and monthly consumption ---
    chart_cols = st.columns(2)
    with chart_cols[0]:
        with st.container(border=True):
            fig = pl.plot_daily(pl.daily_energy(feat))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    with chart_cols[1]:
        with st.container(border=True):
            fig = pl.plot_monthly(pl.monthly_energy(feat))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # --- Row 3: day-type split, hourly profile, peak events, cost (all cards
    # share the same 300px container height, so the two chart cards and the
    # two non-chart cards - which would otherwise be much shorter - line up) ---
    small_cols = st.columns(4)
    with small_cols[0]:
        with st.container(border=True, height=300):
            fig = pl.plot_day_type_donut(behavioral)
            fig.update_layout(height=230)
            st.plotly_chart(fig, use_container_width=True)
    with small_cols[1]:
        with st.container(border=True, height=300):
            fig = pl.plot_hourly_profile(pl.hourly_profile(feat))
            fig.update_layout(height=230)
            st.plotly_chart(fig, use_container_width=True)
    with small_cols[2]:
        with st.container(border=True, height=300):
            st.caption("Peak Events")
            peak_count = peak.get("peak_count")
            st.write(f"**Total Peak Events:** {int(peak_count) if peak_count is not None else '—'}")
            st.write(f"**Avg Peak Magnitude:** {fmt_kw(peak.get('mean_peak'))}")
            st.write(f"**Top Peak:** {fmt_kw(peak.get('max_peak'))}")
    with small_cols[3]:
        with st.container(border=True, height=300):
            if financial:
                render_icon_metric("Estimated Annual Cost", fmt_eur(financial["total_network_cost_eur"]), ICON_MONEY)
            else:
                st.info("No financial data available for this customer.")

    st.divider()
    st.subheader("🩺 Diagnostic summary")
    render_short_diagnosis(st.session_state.diagnosis)

# ------------------------------------------------------------
# CONSUMPTION
# ------------------------------------------------------------
elif page == "Consumption":
    st.title("Consumption")

    behavioral = record["behavioral_features"]
    financial = record["financial"]

    st.subheader("Key Findings")
    cols = st.columns(5)
    with cols[0]:
        with st.container(border=True, height=150):
            render_icon_metric("Annual Consumption", fmt_annual_consumption(behavioral["annual_consumption"]), ICON_BARS)
    with cols[1]:
        with st.container(border=True, height=150):
            render_icon_metric("Average Load", fmt_kw(behavioral["mean_consumption"]), ICON_PULSE)
    with cols[2]:
        with st.container(border=True, height=150):
            render_icon_metric("Peak Load", fmt_kw(behavioral["max_consumption"]), ICON_BOLT)
    with cols[3]:
        with st.container(border=True, height=150):
            mean_consumption = behavioral.get("mean_consumption")
            peak_avg_ratio = behavioral["max_consumption"] / mean_consumption if mean_consumption else None
            render_icon_metric("Peak Average", f"{peak_avg_ratio:.2f}" if peak_avg_ratio is not None else "—", ICON_GAUGE)
    with cols[4]:
        with st.container(border=True, height=150):
            load_factor = behavioral.get("load_factor")
            st.metric("Load Factor", f"{load_factor:.2f}" if load_factor is not None else "—")

    feat = pl.add_time_features(electricity_df)
    view_tabs = st.tabs([
        "Full year", "Monthly", "Weekly", "Daily", "Seasonal", "Weekday vs. weekend",
        "Load Duration Curve", "Peak Events",
    ])

    with view_tabs[0]:
        st.plotly_chart(pl.plot_full_year(electricity_df), use_container_width=True)
    with view_tabs[1]:
        unit = st.radio("Unit", ["kWh", "€"], horizontal=True, label_visibility="collapsed", key="monthly_unit")
        monthly_df = pl.monthly_energy(feat)
        if unit == "kWh":
            st.plotly_chart(pl.plot_monthly(monthly_df), use_container_width=True)
        elif financial:
            monthly_cost_df = pl.monthly_cost(monthly_df, financial["arbeitspreis_ct_kwh"])
            st.plotly_chart(pl.plot_monthly_cost(monthly_cost_df), use_container_width=True)
            st.caption(
                "Energy charge only - excludes the annual, peak-based demand charge "
                "(see Financial Findings below)."
            )
        else:
            st.info("No financial data available - cannot show the cost view.")
    with view_tabs[2]:
        st.plotly_chart(pl.plot_weekly(pl.weekly_energy(feat)), use_container_width=True)
    with view_tabs[3]:
        st.plotly_chart(pl.plot_daily(pl.daily_energy(feat)), use_container_width=True)
    with view_tabs[4]:
        st.plotly_chart(pl.plot_seasonal(pl.seasonal_energy(feat)), use_container_width=True)
    with view_tabs[5]:
        profile_col, totals_col = st.columns(2)
        with profile_col:
            fig = pl.plot_weekday_weekend_profile(pl.weekday_weekend_profile(feat))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with totals_col:
            fig = pl.plot_weekday_weekend_totals(pl.weekday_weekend_totals(feat))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    with view_tabs[6]:
        st.plotly_chart(pl.plot_load_duration_curve(pl.load_duration_curve(electricity_df)), use_container_width=True)
        st.caption("How many hours per year this customer's load runs at or above a given level.")
    with view_tabs[7]:
        peak_events = dx.get_peak_events(electricity_df)
        if not peak_events.empty:
            display_df = peak_events[["peak_time", "peak_value", "duration_hours", "peak_ratio", "is_weekend"]].copy()
            display_df["peak_time"] = pd.to_datetime(display_df["peak_time"]).dt.strftime("%d.%m.%Y %H:%M")
            display_df["peak_value"] = display_df["peak_value"].round(1)
            display_df["duration_hours"] = display_df["duration_hours"].round(2)
            display_df["peak_ratio"] = display_df["peak_ratio"].round(2)
            display_df = display_df.rename(columns={
                "peak_time": "Peak Time", "peak_value": "Peak Value (kW)",
                "duration_hours": "Duration (h)", "peak_ratio": "Peak Ratio", "is_weekend": "Weekend?",
            }).sort_values("Peak Value (kW)", ascending=False).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.caption(f"{len(display_df)} peak events detected - click a column header to sort.")
        else:
            st.info("No peak events detected for this customer.")

    st.divider()
    st.subheader("Financial Findings")

    if financial:
        cols = st.columns(4)
        with cols[0]:
            with st.container(border=True, height=150):
                render_icon_metric("Total Annual Network Cost", fmt_eur(financial["total_network_cost_eur"]), ICON_MONEY)
        with cols[1]:
            with st.container(border=True, height=150):
                st.metric("Demand Charge", fmt_eur(financial["demand_charge_eur"]))
        with cols[2]:
            with st.container(border=True, height=150):
                st.metric("Energy Charge", fmt_eur(financial["energy_charge_eur"]))
        with cols[3]:
            with st.container(border=True, height=150):
                st.metric("Utilization Bracket", financial["utilization_bracket"])

        with st.container(border=True):
            fig = pl.plot_cost_breakdown(financial)
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No financial data available for this customer.")

# ------------------------------------------------------------
# DIAGNOSTIC & STRATEGY
# ------------------------------------------------------------
elif page == "Diagnostic & Strategy":
    st.title("Diagnostic & Strategy")

    st.subheader("Computed peak-reduction scenarios")
    st.caption(
        "Directly computed by the financial engine - these numbers are independent of, and "
        "always available before, any AI-generated text below."
    )
    if record["strategies"]:
        scenarios_df = pd.DataFrame(record["strategies"])[[
            "reduction_pct", "baseline_peak_kw", "new_peak_kw", "savings_eur", "savings_pct",
            "baseline_bracket", "new_bracket", "bracket_changed",
        ]].rename(columns={
            "reduction_pct": "Reduction %", "baseline_peak_kw": "Baseline peak (kW)",
            "new_peak_kw": "New peak (kW)", "savings_eur": "Savings (€)", "savings_pct": "Savings (%)",
            "baseline_bracket": "Baseline bracket", "new_bracket": "New bracket",
            "bracket_changed": "Bracket changed?",
        })
        st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
    else:
        st.info("No financial data available for this customer - scenarios cannot be computed.")

    st.divider()
    st.subheader("AI diagnostic narrative")

    diagnosis = st.session_state.diagnosis
    if diagnosis is None:
        st.info("Not generated yet. Click **Generate diagnosis & strategy** in the sidebar.")
    else:
        st.markdown("#### Profile summary")
        st.write(diagnosis["profile_summary"])

        st.markdown("#### Behavior summary")
        st.write(diagnosis["behavior_summary"])

        st.markdown("#### Main cost driver")
        st.write(diagnosis["main_cost_driver"] or "_Not clearly identified from the available data._")

        st.markdown("#### Key findings")
        for finding in diagnosis["key_findings"]:
            st.markdown(f"- {finding}")

        st.markdown("#### Worth investigating")
        st.caption("General patterns to go check, based on the findings above - not customer-specific facts or costed recommendations.")
        for pointer in diagnosis["investigation_pointers"]:
            st.markdown(f"- {pointer}")

        st.markdown("#### Consultant focus")
        for topic in diagnosis["consultant_focus"]:
            st.markdown(f"- {topic}")

        st.markdown("#### Caveats")
        for caveat in diagnosis["caveats"]:
            st.warning(caveat)
