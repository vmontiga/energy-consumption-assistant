from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PRIMARY = "#22C55E"          # Light Green / Success, from the EASI color map - main line/bar color
PRIMARY_FILL = "rgba(34, 197, 94, 0.22)"   # translucent Light Green, for area fills under lines
SECONDARY = "#14A06B"        # Green, from the EASI color map - second category in two-way contrasts
WEEKDAY_COLOR = PRIMARY
WEEKEND_COLOR = SECONDARY

BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",   # transparent - blends into the app's dark page
    plot_bgcolor="rgba(0,0,0,0)",    # instead of sitting in its own boxed-in rectangle
    hovermode="x unified",
    margin=dict(l=40, r=20, t=50, b=40),
    font=dict(color="#FAFAFA"),
    xaxis=dict(gridcolor="rgba(30, 77, 66, 0.6)"),   # Border/Divider color, translucent
    yaxis=dict(gridcolor="rgba(30, 77, 66, 0.6)"),
)

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
# Four shades of green rather than four unrelated hues - keeps the whole
# app's chart palette to one color family, while shade variation still
# lets the four categories read as distinct.
# Four of the five EASI primary brand colors, assigned by "temperature" -
# bright/warm-feeling greens for warm seasons, the darkest green for winter -
# rather than four unrelated hues.
SEASON_COLORS = {"Spring": "#A3E635", "Summer": "#D9F99D", "Autumn": "#14A06B", "Winter": "#0A3D2E"}


# ============================================================
# DATA PREPARATION
# ============================================================

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp_local") -> pd.DataFrame:
    """Add calendar/time features used across the various views."""
    out = df.copy()
    ts = out[timestamp_col]
    out["date"] = ts.dt.date
    out["year"] = ts.dt.year
    out["month"] = ts.dt.month
    out["month_name"] = ts.dt.strftime("%b")
    out["iso_week"] = ts.dt.isocalendar().week.values
    out["weekday"] = ts.dt.weekday
    out["weekday_name"] = ts.dt.strftime("%A")
    out["is_weekend"] = out["weekday"] >= 5
    out["day_type"] = out["is_weekend"].map({True: "Weekend", False: "Weekday"})
    out["season"] = out["month"].map(SEASON_MAP)
    out["hour_decimal"] = ts.dt.hour + ts.dt.minute / 60.0
    return out


def monthly_energy(df_feat: pd.DataFrame, value_col: str = "power_kw", interval_hours: float = 0.25) -> pd.DataFrame:
    g = (
        df_feat.assign(energy_kwh=df_feat[value_col] * interval_hours)
        .groupby(["year", "month", "month_name"], as_index=False)["energy_kwh"]
        .sum()
        .sort_values(["year", "month"])
    )
    g["label"] = g["month_name"] + " " + g["year"].astype(str)
    return g


def weekly_energy(df_feat: pd.DataFrame, value_col: str = "power_kw", interval_hours: float = 0.25) -> pd.DataFrame:
    g = (
        df_feat.assign(energy_kwh=df_feat[value_col] * interval_hours)
        .groupby(["year", "iso_week"], as_index=False)["energy_kwh"]
        .sum()
        .sort_values(["year", "iso_week"])
    )
    g["label"] = "Week " + g["iso_week"].astype(str)
    return g


def daily_energy(df_feat: pd.DataFrame, value_col: str = "power_kw", interval_hours: float = 0.25) -> pd.DataFrame:
    g = (
        df_feat.assign(energy_kwh=df_feat[value_col] * interval_hours)
        .groupby("date", as_index=False)["energy_kwh"]
        .sum()
        .sort_values("date")
    )
    g["date"] = pd.to_datetime(g["date"])
    return g


def seasonal_energy(df_feat: pd.DataFrame, value_col: str = "power_kw", interval_hours: float = 0.25) -> pd.DataFrame:
    g = (
        df_feat.assign(energy_kwh=df_feat[value_col] * interval_hours)
        .groupby("season", as_index=False)["energy_kwh"]
        .sum()
    )
    g["season"] = pd.Categorical(g["season"], categories=SEASON_ORDER, ordered=True)
    return g.sort_values("season").reset_index(drop=True)


def weekday_weekend_profile(df_feat: pd.DataFrame, value_col: str = "power_kw") -> pd.DataFrame:
    """Average power by time-of-day, split weekday vs weekend - the
    standard 'typical load profile' comparison."""
    return (
        df_feat.groupby(["day_type", "hour_decimal"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "power_kw"})
        .sort_values(["day_type", "hour_decimal"])
    )


def weekday_weekend_totals(df_feat: pd.DataFrame, value_col: str = "power_kw", interval_hours: float = 0.25) -> pd.DataFrame:
    """Average daily energy (kWh/day) for weekdays vs weekend days -
    normalized so the (usually 5 vs 2 day) imbalance doesn't skew the comparison."""
    per_day = (
        df_feat.assign(energy_kwh=df_feat[value_col] * interval_hours)
        .groupby(["date", "day_type"], as_index=False)["energy_kwh"].sum()
    )
    g = per_day.groupby("day_type", as_index=False)["energy_kwh"].mean()
    return g.rename(columns={"energy_kwh": "avg_daily_energy_kwh"})


def hourly_profile(df_feat: pd.DataFrame, value_col: str = "power_kw") -> pd.DataFrame:
    """Average power by time-of-day, across ALL days (weekday and weekend
    combined) - a single overall daily curve, as opposed to
    weekday_weekend_profile's split-by-day-type version."""
    return (
        df_feat.groupby("hour_decimal", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "power_kw"})
        .sort_values("hour_decimal")
    )


def load_duration_curve(df: pd.DataFrame, value_col: str = "power_kw", interval_hours: float = 0.25) -> pd.DataFrame:
    """Every reading sorted from highest to lowest, against the cumulative
    hours per year at or above that level - the standard energy-analysis
    'load duration curve'. Answers 'how many hours a year does this
    customer run above X kW', and connects directly to the utilization-hours
    concept that drives the tariff bracket in the financial calculation
    (see financial.py)."""
    sorted_values = df[value_col].sort_values(ascending=False).reset_index(drop=True)
    hours = (sorted_values.index + 1) * interval_hours
    return pd.DataFrame({"hours": hours, "power_kw": sorted_values.values})


def monthly_cost(monthly_df: pd.DataFrame, arbeitspreis_ct_kwh: float) -> pd.DataFrame:
    """Monthly ENERGY CHARGE ONLY (using the tariff's per-kWh rate) - not
    total monthly cost. The demand charge is an annual, peak-based figure
    under this project's tariff structure (Jahresleistungspreissystem,
    see pricing.py) and has no physically meaningful monthly decomposition -
    inventing one would violate this project's 'never invent a financial
    number' principle. This is the energy-charge portion only, clearly
    labeled as such wherever it's shown."""
    out = monthly_df.copy()
    out["cost_eur"] = out["energy_kwh"] * (arbeitspreis_ct_kwh / 100)
    return out


# ============================================================
# CHARTS
# ============================================================

def plot_full_year(df: pd.DataFrame, timestamp_col: str = "timestamp_local", value_col: str = "power_kw") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[timestamp_col], y=df[value_col], mode="lines",
        line=dict(color=PRIMARY, width=1), fill="tozeroy", fillcolor=PRIMARY_FILL,
        name="Power (kW)",
    ))
    fig.update_layout(
        title="Full Year - Power Profile", xaxis_title="Time", yaxis_title="Power (kW)",
        xaxis_rangeslider_visible=True, **BASE_LAYOUT,
    )
    return fig


def _bar(labels: list[str], values: pd.Series, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=PRIMARY, name=y_title))
    fig.update_layout(title=title, yaxis_title=y_title, **BASE_LAYOUT)
    return fig


def plot_monthly(monthly_df: pd.DataFrame) -> go.Figure:
    return _bar(monthly_df["label"].tolist(), monthly_df["energy_kwh"], "Monthly Consumption", "Energy (kWh)")


def plot_weekly(weekly_df: pd.DataFrame) -> go.Figure:
    fig = _bar(weekly_df["label"].tolist(), weekly_df["energy_kwh"], "Weekly Consumption", "Energy (kWh)")
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    return fig


def plot_daily(daily_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df["date"], y=daily_df["energy_kwh"], mode="lines",
        line=dict(color=PRIMARY, width=1.5), fill="tozeroy", fillcolor=PRIMARY_FILL,
        name="Energy (kWh)",
    ))
    fig.update_layout(
        title="Daily Consumption Over the Year", xaxis_title="Date", yaxis_title="Energy (kWh)",
        xaxis_rangeslider_visible=True, **BASE_LAYOUT,
    )
    return fig


def plot_seasonal(seasonal_df: pd.DataFrame) -> go.Figure:
    values = seasonal_df["energy_kwh"]
    labels = seasonal_df["season"].astype(str).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=[SEASON_COLORS.get(s, PRIMARY) for s in labels], name="Energy (kWh)"))
    fig.update_layout(title="Seasonal Consumption", yaxis_title="Energy (kWh)", **BASE_LAYOUT)
    return fig


def plot_weekday_weekend_profile(profile_df: pd.DataFrame) -> go.Figure:
    """Two overlaid average daily load curves: weekday vs weekend."""
    fig = go.Figure()
    for day_type, color in [("Weekday", WEEKDAY_COLOR), ("Weekend", WEEKEND_COLOR)]:
        sub = profile_df[profile_df["day_type"] == day_type]
        hh = (sub["hour_decimal"] // 1).astype(int)
        mm = ((sub["hour_decimal"] % 1) * 60).astype(int)
        x_labels = [f"{h:02d}:{m:02d}" for h, m in zip(hh, mm)]
        fig.add_trace(go.Scatter(
            x=x_labels, y=sub["power_kw"], mode="lines",
            line=dict(color=color, width=2), name=day_type,
        ))

    fig.update_layout(
        title="Typical Load Profile: Weekday vs. Weekend", xaxis_title="Time of day", yaxis_title="Avg power (kW)",
        **BASE_LAYOUT,
    )
    fig.update_xaxes(tickmode="array", tickvals=[f"{h:02d}:00" for h in range(0, 24, 2)])
    return fig


def plot_weekday_weekend_totals(totals_df: pd.DataFrame) -> go.Figure:
    fig = _bar(
        totals_df["day_type"].tolist(), totals_df["avg_daily_energy_kwh"],
        "Weekday vs. Weekend - Avg Daily Consumption", "Avg energy per day (kWh)",
    )
    fig.data[0].marker.color = [WEEKDAY_COLOR if d == "Weekday" else WEEKEND_COLOR for d in totals_df["day_type"]]
    return fig


def plot_hourly_profile(profile_df: pd.DataFrame) -> go.Figure:
    """Single averaged daily load curve across all days (weekday and
    weekend combined) - a simpler alternative to
    plot_weekday_weekend_profile's split-by-day-type version, used on the
    Dashboard tab."""
    hh = (profile_df["hour_decimal"] // 1).astype(int)
    mm = ((profile_df["hour_decimal"] % 1) * 60).astype(int)
    x_labels = [f"{h:02d}:{m:02d}" for h, m in zip(hh, mm)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=profile_df["power_kw"], mode="lines", line=dict(color=PRIMARY, width=2)))
    fig.update_layout(title="Hourly Profile (Average)", xaxis_title="Time of day", yaxis_title="Avg power (kW)", **BASE_LAYOUT)
    fig.update_xaxes(tickmode="array", tickvals=[f"{h:02d}:00" for h in range(0, 24, 4)])
    return fig


def plot_load_duration_curve(duration_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=duration_df["hours"], y=duration_df["power_kw"], mode="lines",
        line=dict(color=PRIMARY, width=1.5), fill="tozeroy", fillcolor=PRIMARY_FILL,
    ))
    fig.update_layout(
        title="Load Duration Curve", xaxis_title="Hours per year at or above this level",
        yaxis_title="Power (kW)", **BASE_LAYOUT,
    )
    return fig


def plot_monthly_cost(monthly_cost_df: pd.DataFrame) -> go.Figure:
    """Monthly energy charge (see monthly_cost() - energy charge only, not
    total cost)."""
    fig = _bar(monthly_cost_df["label"].tolist(), monthly_cost_df["cost_eur"], "Monthly Energy Charge", "EUR")
    return fig


# ============================================================
# DASHBOARD-SPECIFIC CHARTS (aggregate summaries, not time series)
# ============================================================

def plot_time_of_day_shares(behavioral_features: dict) -> go.Figure:
    """Horizontal bar chart of annual consumption share by time-of-day
    window."""
    labels = ["Night (00-06)", "Morning (06-10)", "Daytime (10-17)", "Evening (17-22)"]
    keys = ["night_share", "morning_share", "daytime_share", "evening_share"]
    values = [(behavioral_features.get(k) or 0) * 100 for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker_color=PRIMARY))
    fig.update_layout(
        title="Consumption Share by Time of Day", xaxis_title="Share of annual consumption (%)",
        **BASE_LAYOUT,
    )
    return fig


def plot_day_type_donut(behavioral_features: dict) -> go.Figure:
    """Donut chart of annual consumption share: weekday vs. weekend - used
    on the Dashboard tab. An intentional exception to preferring bars
    elsewhere in this project: with exactly two categories that must sum
    to 100%, a donut reads faster than a two-bar chart would."""
    values = [(behavioral_features.get("weekday_share") or 0) * 100, (behavioral_features.get("weekend_share") or 0) * 100]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["Weekdays", "Weekends"], values=values, hole=0.6,
        marker=dict(colors=[WEEKDAY_COLOR, WEEKEND_COLOR]),
        textinfo="label+percent", textfont=dict(color="#FAFAFA"),
    ))
    fig.update_layout(title="Consumption by Day Type", showlegend=False, **BASE_LAYOUT)
    return fig


def plot_cost_breakdown(financial: dict) -> go.Figure:
    """Horizontal bar chart splitting annual network cost into demand vs.
    energy charge. Caller should only call this when `financial` is not
    None."""
    labels = ["Demand charge (Leistungspreis)", "Energy charge (Arbeitspreis)"]
    values = [financial.get("demand_charge_eur") or 0, financial.get("energy_charge_eur") or 0]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker_color=[PRIMARY, SECONDARY]))
    fig.update_layout(title="Annual Network Cost Breakdown", xaxis_title="EUR / year", **BASE_LAYOUT)
    return fig
