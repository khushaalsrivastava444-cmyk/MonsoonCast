"""Streamlit Research Page 2: Model Comparison & Verification."""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Comparison | MonsoonCast", page_icon="📊", layout="wide")

st.title("📊 Model Verification & Performance Comparison")
st.markdown("Comparative evaluation of baselines, machine learning models, and calibrated ensembles on held-out test data.")

results_path = Path("artifacts/metrics/evaluation_results.json")

if not results_path.exists():
    st.warning("Evaluation results not found. Please run 'python scripts/evaluate.py' to generate metrics.")
    st.stop()

with open(results_path, "r") as f:
    eval_data = json.load(f)

# Model Comparison Table
st.subheader("1. Comprehensive Model Verification Benchmark")

rows = []
for m_name, m_dict in eval_data.items():
    if "pr_auc" in m_dict:
        ci = m_dict.get("csi_95ci", [0, 0])
        rows.append({
            "Model": m_name,
            "PR-AUC": m_dict.get("pr_auc"),
            "ROC-AUC": m_dict.get("roc_auc"),
            "CSI (Threat Score)": m_dict.get("csi"),
            "CSI 95% CI": f"[{ci[0]:.2f}, {ci[1]:.2f}]",
            "POD (Recall)": m_dict.get("pod"),
            "FAR (False Alarm)": m_dict.get("far"),
            "Brier Score": m_dict.get("brier_score"),
            "BSS (Skill vs Climatology)": m_dict.get("brier_skill_score"),
            "ECE": m_dict.get("ece")
        })

df_comp = pd.DataFrame(rows)
st.dataframe(
    df_comp.style.highlight_max(subset=["PR-AUC", "ROC-AUC", "CSI (Threat Score)", "BSS (Skill vs Climatology)"], color="#c6f6d5")
           .highlight_min(subset=["FAR (False Alarm)", "Brier Score", "ECE"], color="#bee3f8"),
    use_container_width=True
)

st.markdown("---")

# Comparative Bar Chart
st.subheader("2. Metric Comparison Across Model Architectures")
metric_choice = st.selectbox(
    "Select Verification Metric to Compare",
    options=["CSI (Threat Score)", "BSS (Skill vs Climatology)", "PR-AUC", "POD (Recall)", "FAR (False Alarm)", "ECE"]
)

fig_bar = px.bar(
    df_comp,
    x="Model",
    y=metric_choice,
    color="Model",
    title=f"Comparison of {metric_choice} across Models",
    text_auto=".3f",
    color_discrete_sequence=px.colors.qualitative.Prism
)
fig_bar.update_layout(template="plotly_white", showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# Trade-off Analysis: POD vs FAR
st.subheader("3. Extreme Event Trade-off: Detection vs False Alarms")
col_a, col_b = st.columns(2)

with col_a:
    fig_scatter = px.scatter(
        df_comp,
        x="FAR (False Alarm)",
        y="POD (Recall)",
        text="Model",
        size="CSI (Threat Score)",
        color="Model",
        title="POD vs FAR (Upper Left is Ideal)",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_b:
    st.markdown("""
    ### Verification Insights:
    - **Climatology Baseline:** High reliability on average, but zero capability to discriminate localized extreme days (low CSI).
    - **Persistence Baseline:** Relies purely on $t-1$ rainfall; misses synoptic developments and has high false alarms after convective decay.
    - **Random Forest & XGBoost:** Strong discriminatory power (high PR-AUC), but raw probabilities often require calibration.
    - **Calibrated Ensemble:** Achieves optimal Brier Skill Score (BSS) and lowest Expected Calibration Error (ECE), proving ensembling improves probabilistic forecast quality.
    """)
