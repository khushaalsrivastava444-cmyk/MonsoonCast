"""Streamlit Research Page 2: Model Comparison & Verification."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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

# Section 4: Feature Importance & Sensitivity
st.markdown("---")
st.subheader("4. Meteorological Predictor Importance & Sensitivity Analysis")

feat_path = Path("artifacts/metrics/feature_importance.json")
sens_path = Path("artifacts/metrics/sensitivity_analysis.json")

tab_feat, tab_sens = st.tabs(["Meteorological Predictor Importance", "Threshold & Regional Sensitivity"])

with tab_feat:
    if feat_path.exists():
        with open(feat_path, "r") as f:
            feat_data = json.load(f)
        if "permutation_importance" in feat_data:
            p_imp = feat_data["permutation_importance"]
            df_feat = pd.DataFrame([
                {"Predictor": k, "Permutation Drop in AUC": v["mean_delta_auc"], "Std": v["std_delta_auc"]}
                for k, v in p_imp.items()
            ]).sort_values("Permutation Drop in AUC", ascending=False)

            fig_feat = px.bar(
                df_feat.head(12),
                x="Permutation Drop in AUC",
                y="Predictor",
                orientation="h",
                title="Top Predictors by Permutation Importance (Test Set)",
                color="Permutation Drop in AUC",
                color_continuous_scale="Blues"
            )
            fig_feat.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
            st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Run 'python scripts/analyze_feature_importance.py' to generate feature importance.")

with tab_sens:
    if sens_path.exists():
        with open(sens_path, "r") as f:
            sens_data = json.load(f)
        t_data = sens_data.get("threshold_sensitivity", {})
        if t_data:
            df_t = pd.DataFrame([
                {
                    "Threshold (mm/day)": v["threshold_mm"],
                    "Classification": v["label"],
                    "Prevalence (%)": v["event_prevalence_pct"],
                    "ROC-AUC": v["roc_auc"],
                    "CSI": v["csi"],
                    "ECE": v["ece"]
                }
                for v in t_data.values()
            ])
            st.markdown("#### Rainfall Threshold Sensitivity (IMD Categories)")
            st.dataframe(df_t, use_container_width=True)

            fig_t = px.line(
                df_t,
                x="Threshold (mm/day)",
                y=["ROC-AUC", "CSI"],
                markers=True,
                title="Model Performance Across Extreme Rainfall Thresholds"
            )
            fig_t.update_layout(template="plotly_white")
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Run 'python scripts/sensitivity_analysis.py' to generate sensitivity analysis.")

