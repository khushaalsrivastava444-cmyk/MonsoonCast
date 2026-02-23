"""Streamlit Research Page 1: Rainfall Explorer & Climatology."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Rainfall Explorer | MonsoonCast", page_icon="🌧️", layout="wide")

st.title("🌧️ Monsoon Rainfall Explorer & Spatio-Temporal Climatology")
st.markdown("Exploratory analysis of Summer Monsoon (June–September) precipitation fields over Central India.")

test_path = Path("data/processed/test.parquet")
train_path = Path("data/processed/train.parquet")

if not test_path.exists() and not train_path.exists():
    st.warning("Processed dataset not found. Please run 'python scripts/prepare_data.py' to process the data.")
    st.stop()

@st.cache_data
def load_data():
    df_t = pd.read_parquet(train_path) if train_path.exists() else pd.DataFrame()
    df_e = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()
    return pd.concat([df_t, df_e], axis=0, ignore_index=True)

df = load_data()

# Sidebar Controls
st.sidebar.header("Explorer Controls")
threshold = st.sidebar.slider("Extreme Event Threshold (mm/day)", min_value=25.0, max_value=150.0, value=50.0, step=5.0)
year_options = sorted(df["year"].unique()) if "year" in df.columns else []
selected_year = st.sidebar.selectbox("Filter Year", options=year_options, index=len(year_options)-1 if year_options else 0)

# Summary Stats
st.subheader("1. Observational Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

df_year = df[df["year"] == selected_year] if selected_year else df
rain_vals = df_year["rain_current"].values
extreme_count = np.sum(rain_vals >= threshold)
total_count = len(rain_vals)
pct_extreme = (extreme_count / total_count) * 100.0 if total_count > 0 else 0.0

with col1:
    st.metric("Total Grid-Days", f"{total_count:,}")
with col2:
    st.metric("Mean Daily Rainfall", f"{np.mean(rain_vals):.2f} mm/day")
with col3:
    st.metric("Max Daily Rainfall", f"{np.max(rain_vals):.2f} mm/day")
with col4:
    st.metric(f"Events >= {threshold:.0f} mm", f"{extreme_count:,} ({pct_extreme:.2f}%)")

st.markdown("---")

# Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Monsoon Rainfall Distribution")
    fig_hist = px.histogram(
        df_year[df_year["rain_current"] > 0.1],
        x="rain_current",
        nbins=50,
        log_y=True,
        title=f"Precipitation Distribution on Wet Days ({selected_year}) [Log Scale]",
        labels={"rain_current": "Daily Rainfall (mm/day)"},
        color_discrete_sequence=["#1f77b4"]
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold} mm)")
    fig_hist.update_layout(template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.subheader("Seasonal Progression (JJAS)")
    df_doy = df_year.groupby("doy")["rain_current"].agg(["mean", "max"]).reset_index()
    fig_seas = go.Figure()
    fig_seas.add_trace(go.Scatter(x=df_doy["doy"], y=df_doy["mean"], mode="lines", name="Spatial Mean Rainfall", line=dict(color="#2b6cb0", width=2)))
    fig_seas.add_trace(go.Scatter(x=df_doy["doy"], y=df_doy["max"], mode="lines", name="Spatial Peak Rainfall", line=dict(color="#e53e3e", dash="dot")))
    fig_seas.update_layout(
        title=f"Daily Monsoon Rainfall Evolution across Season ({selected_year})",
        xaxis_title="Day of Year (DOY)",
        yaxis_title="Rainfall (mm/day)",
        template="plotly_white"
    )
    st.plotly_chart(fig_seas, use_container_width=True)

# Spatial Heatmap
st.subheader("2. Spatial Rainfall Field")
available_dates = df_year["time"].dt.strftime("%Y-%m-%d").unique()
date_select = st.select_slider("Select Date to View Spatial Field", options=list(available_dates))

df_date = df_year[df_year["time"].dt.strftime("%Y-%m-%d") == date_select]

fig_map = px.density_mapbox(
    df_date,
    lat="lat",
    lon="lon",
    z="rain_current",
    radius=30,
    center=dict(lat=float(df_date["lat"].mean()), lon=float(df_date["lon"].mean())),
    zoom=5,
    mapbox_style="carto-positron",
    title=f"Central India Gridded Rainfall Field on {date_select}",
    color_continuous_scale="Blues",
    range_color=[0, max(50.0, float(df_date["rain_current"].max()))]
)
fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
st.plotly_chart(fig_map, use_container_width=True)
