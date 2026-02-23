"""Streamlit Research Page 3: Uncertainty Quantification & Calibration Analysis."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Uncertainty Analysis | MonsoonCast", page_icon="🎯", layout="wide")

st.title("🎯 Uncertainty Quantification & Probability Calibration")
st.markdown("Rigorous assessment of probabilistic calibration, reliability curves, ensemble spread, and quantile prediction intervals.")

rel_path = Path("artifacts/metrics/reliability_curves.json")
results_path = Path("artifacts/metrics/evaluation_results.json")

if not rel_path.exists():
    st.warning("Reliability curve data not found. Please run 'python scripts/evaluate.py' to generate metrics.")
    st.stop()

with open(rel_path, "r") as f:
    rel_curves = json.load(f)

with open(results_path, "r") as f:
    eval_results = json.load(f)

# 1. Multi-Model Reliability Diagrams
st.subheader("1. Reliability Diagrams (Calibration Curves)")
st.markdown("A calibrated probabilistic model produces predictions where a forecast of $p$ matches the empirical event frequency $p$.")

col1, col2 = st.columns([1.5, 1])

with col1:
    fig_rel = go.Figure()

    # 45-degree reference line
    fig_rel.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray", width=2),
            name="Perfect Calibration"
        )
    )

    colors = {
        "climatology": "#718096",
        "persistence": "#e53e3e",
        "logistic_regression": "#d69e2e",
        "random_forest": "#3182ce",
        "xgboost_classifier": "#38a169",
        "ensemble": "#805ad5"
    }

    for m_name, rel_data in rel_curves.items():
        bin_means = rel_data["bin_means"]
        bin_fracs = rel_data["bin_fracs"]
        ece = rel_data.get("ece", 0.0)

        fig_rel.add_trace(
            go.Scatter(
                x=bin_means,
                y=bin_fracs,
                mode="lines+markers",
                name=f"{m_name} (ECE={ece:.3f})",
                line=dict(color=colors.get(m_name, "#4a5568"), width=2.5),
                marker=dict(size=7)
            )
        )

    fig_rel.update_layout(
        title="Reliability Curves on Held-Out Test Set",
        xaxis_title="Forecast Probability",
        yaxis_title="Observed Relative Frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=520
    )
    st.plotly_chart(fig_rel, use_container_width=True)

with col2:
    st.markdown("### Expected Calibration Error (ECE):")
    ece_list = []
    for m_name, rel_data in rel_curves.items():
        ece_list.append({"Model": m_name, "ECE": rel_data.get("ece", 0.0)})
    df_ece = pd.DataFrame(ece_list).sort_values("ECE")
    st.dataframe(df_ece, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Models lying **above the diagonal** are *underconfident* (events occur more frequently than predicted).
    - Models lying **below the diagonal** are *overconfident* (forecast high probability for events that fail to materialize).
    - The **Calibrated Ensemble** brings probability forecasts closest to the diagonal across the critical high-risk probability spectrum.
    """)

st.markdown("---")

# 2. Prediction Intervals and Quantile Coverage
st.subheader("2. Conditional Rainfall Prediction Intervals (Quantile Regression)")
if "quantile_regressor" in eval_results:
    q_dict = eval_results["quantile_regressor"]
    col_q1, col_q2, col_q3 = st.columns(3)

    with col_q1:
        st.metric("Nominal Interval Coverage", "80.0%", "10th to 90th Quantile")
    with col_q2:
        picp = q_dict.get("picp", 0.0) * 100.0
        st.metric("Observed Empirical Coverage (PICP)", f"{picp:.1f}%")
    with col_q3:
        mpiw = q_dict.get("mpiw", 0.0)
        st.metric("Mean Prediction Interval Width (MPIW)", f"{mpiw:.2f} mm")
else:
    st.info("Quantile regression evaluation will appear here once trained.")

st.markdown("---")

# 3. Ensemble Epistemic Uncertainty
st.subheader("3. Ensemble Member Spread & Disagreement")
st.markdown("""
During synoptic monsoon depressions, convective initiation exhibits strong chaotic sensitivity.
The ensemble spread $\\sigma = \\sqrt{\frac{1}{K}\\sum(p_k - \bar{p})^2}$ provides a physically motivated metric of epistemic uncertainty, signaling when the operational forecast carries elevated risk of severe prediction error.
""")
