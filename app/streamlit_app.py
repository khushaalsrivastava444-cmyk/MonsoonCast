"""MonsoonCast: Research-Grade Streamlit Research Dashboard.

An AI Ensemble Framework for Extreme Monsoon Rainfall Prediction and Uncertainty Quantification over India.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="MonsoonCast | AI Research Framework",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f2b48;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .callout {
        padding: 0.8rem 1.2rem;
        background-color: #edf2f7;
        border-left: 4px solid #2b6cb0;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">MonsoonCast: Extreme Monsoon Rainfall AI Framework</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Research-Grade Probabilistic Machine Learning & Uncertainty Quantification over India</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="callout">
<b>Primary Research Objective:</b> Evaluating whether physics-informed machine learning ensembles can reliably forecast extreme Indian Summer Monsoon (JJAS) precipitation events, improve probability calibration, and quantify predictive uncertainty compared with classical climatological and statistical baselines.
</div>
""", unsafe_allow_html=True)

# System Status and Artifact Verification
col1, col2, col3, col4 = st.columns(4)

models_path = Path("artifacts/models/monsooncast_models.joblib")
metrics_path = Path("artifacts/metrics/evaluation_results.json")
test_path = Path("data/processed/test.parquet")
synth_sample = Path("data/raw/synthetic_sample.nc")
imd_files = list(Path("data/raw/imd").glob("*.nc")) if Path("data/raw/imd").exists() else []
is_synthetic = len(imd_files) == 0

with col1:
    state_label = "Ready" if models_path.exists() and metrics_path.exists() else "Setup Required"
    st.metric("System State", state_label, "Release v0.3.0 (Feb 2026)")
with col2:
    status_art = "Trained & Available" if models_path.exists() else "Not Trained"
    st.metric("Model Registry", status_art)
with col3:
    status_met = "Available" if metrics_path.exists() else "Pending Run"
    st.metric("Evaluation Metrics", status_met)
with col4:
    grid_str = "0.50° Grid (Synthetic)" if is_synthetic else "0.25° IMD Observed"
    st.metric("Active Domain", "Central India", grid_str)

if is_synthetic:
    st.warning(
        "⚠️ **PROVISIONAL BENCHMARK NOTICE:** The current model artifacts and evaluation metrics were generated "
        "using a deterministic synthetic meteorological test fixture (0.50° resolution). This fixture verifies pipeline "
        "functionality, causal zero-leakage constraints, and calibration algorithms, but must NOT be cited as empirical "
        "meteorological evidence. Download real IMD gridded NetCDF data to data/raw/imd/ for production analysis."
    )

st.markdown("---")

# Executive Overview Tabs
tab_overview, tab_architecture, tab_governance = st.tabs(["Research Overview", "Model Architecture", "Data Governance & Traceability"])

with tab_overview:
    st.subheader("Key Scientific Findings & Methodological Framework")

    col_a, col_b = st.columns([1.5, 1])

    with col_a:
        st.markdown(r"""
        ### Research Questions Addressed:
        1. **Predictive Skill over Baselines:** Does gradient boosting and non-linear ensembling provide statistically significant gains in Critical Success Index (CSI) and Brier Skill Score (BSS) over climatological and persistence priors?
        2. **Probability Calibration:** Can validation-fitted isotonic regression and Platt scaling resolve severe model overconfidence on rare extreme events ($R \ge 50$ mm/day)?
        3. **Predictive Uncertainty:** Does inter-model spread accurately identify regions of high epistemic uncertainty during complex monsoon depressions?
        4. **Quantile Coverage:** Do conditional quantile gradient-boosted trees yield calibrated 80% prediction intervals for continuous rainfall amounts?
        """)

    with col_b:
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            st.markdown("### Published Evaluation Highlights:")
            df_summary = []
            for m_name, m_vals in metrics_data.items():
                if "pr_auc" in m_vals:
                    df_summary.append({
                        "Model": m_name,
                        "PR-AUC": m_vals.get("pr_auc"),
                        "CSI": m_vals.get("csi"),
                        "BSS": m_vals.get("brier_skill_score"),
                        "ECE": m_vals.get("ece")
                    })
            if df_summary:
                st.dataframe(pd.DataFrame(df_summary), use_container_width=True)
        else:
            st.info("Evaluation metrics pending. Run 'python scripts/run_experiment.py' to generate.")

with tab_architecture:
    st.subheader("End-to-End Pipeline Architecture")
    st.code("""
    IMD 0.25 Gridded NetCDF -> Spatial/Monsoon Subsetting (JJAS)
      -> Backward-Shifted Lag & Rolling Feature Extraction
        -> Strict Chronological Split (Train: 2010-2018 | Val: 2019-2020 | Test: 2021-2023)
          -> Baseline Models: Climatology, Persistence, Logistic Regression
          -> Machine Learning: Random Forest, XGBoost Classifier, Quantile Regressors
          -> Weighted Probabilistic Ensemble (Validation-optimized weights + Isotonic Calibration)
            -> Verification: POD, FAR, CSI, ETS, Brier Skill Score, Block Bootstrap CIs
              -> Streamlit Research App & FastAPI Service
    """, language="text")

with tab_governance:
    st.subheader("Scientific Integrity and Data Provenance")
    st.markdown("""
    - **No Temporal Leakage:** Partitioning is performed strictly by calendar years. All feature standardizers, rolling statistics, and calibrators are fitted strictly on training/validation partitions.
    - **Physical Units:** IMD rainfall is verified in millimeters per day (`mm/day`).
    - **Causality Enforcement:** Rolling features at issue time $t$ predict target at $t+1$. Forward rolling windows are strictly prohibited.
    - **Synthetic Disclaimers:** When large observational NetCDFs are not yet downloaded locally, test fixtures are strictly labeled `is_synthetic = True` and never reported as empirical truth.
    """)

st.markdown("---")
st.caption("MonsoonCast Framework | Principal Investigator: Khushaal Srivastava | Released: February 23, 2026")
