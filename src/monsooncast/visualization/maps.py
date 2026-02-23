"""Scientific visualization tools for spatial rainfall fields, uncertainty, and reliability diagrams."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_spatial_rainfall_field(
    df: pd.DataFrame, date: str, val_col: str = "rain_current", title: str | None = None
) -> go.Figure:
    """Plot an interactive spatial scatter / contour heatmap over India coordinates using Plotly."""
    sub = df[df["time"].astype(str).str.startswith(date)].copy()
    if len(sub) == 0:
        sub = df.iloc[:100].copy()
        date = str(sub["time"].iloc[0])[:10]

    fig = px.density_mapbox(
        sub,
        lat="lat",
        lon="lon",
        z=val_col,
        radius=25,
        center=dict(lat=float(sub["lat"].mean()), lon=float(sub["lon"].mean())),
        zoom=4.5,
        mapbox_style="open-street-map",
        title=title or f"Monsoon Rainfall Field ({val_col}) - {date}",
        color_continuous_scale="Blues",
        range_color=[0, max(50.0, float(sub[val_col].max()))],
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def plot_probability_map(
    df: pd.DataFrame,
    prob_col: str = "calibrated_prob",
    title: str = "Predicted Extreme Rainfall Probability",
) -> go.Figure:
    """Plot spatial probability field with threshold markings."""
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color=prob_col,
        size=prob_col,
        color_continuous_scale="Viridis",
        range_color=[0.0, 1.0],
        size_max=12,
        zoom=4.5,
        center=dict(lat=float(df["lat"].mean()), lon=float(df["lon"].mean())),
        mapbox_style="open-street-map",
        title=title,
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def plot_reliability_curve(rel_data: dict[str, Any], model_name: str = "Model") -> go.Figure:
    """Plot scientific reliability diagram (calibration curve) with reference diagonal."""
    fig = go.Figure()

    # Perfectly calibrated reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray", width=2),
            name="Perfect Calibration",
        )
    )

    # Model reliability curve
    bin_means = rel_data["bin_means"]
    bin_fracs = rel_data["bin_fracs"]

    fig.add_trace(
        go.Scatter(
            x=bin_means,
            y=bin_fracs,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=8),
            name=f"{model_name} (ECE={rel_data.get('ece', 0.0):.3f})",
        )
    )

    fig.update_layout(
        title=f"Reliability Diagram — {model_name}",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Empirical Frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        width=600,
        height=500,
    )
    return fig


def plot_precision_recall_curves(
    curves: dict[str, Tuple[np.ndarray, np.ndarray, float]],
) -> go.Figure:
    """Plot multi-model Precision-Recall curves."""
    fig = go.Figure()

    for name, (recalls, precisions, pr_auc) in curves.items():
        fig.add_trace(
            go.Scatter(x=recalls, y=precisions, mode="lines", name=f"{name} (PR-AUC={pr_auc:.3f})")
        )

    fig.update_layout(
        title="Precision-Recall Curves for Extreme Rainfall",
        xaxis_title="Recall (Probability of Detection)",
        yaxis_title="Precision (1 - False Alarm Ratio)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    return fig
