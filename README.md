<div align="center">

# MonsoonCast 🌧️
### An Independent Machine Learning Framework for Extreme Monsoon Rainfall Prediction and Uncertainty Estimation over India

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release: v0.3.0](https://img.shields.io/badge/release-v0.3.0-orange.svg)](CITATION.cff)
[![CI Status](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/API-Live%20on%20Render-009688.svg)](https://monsooncast-api.onrender.com/docs)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://monsooncast-popwidywnbpu3rgwkn5skh.streamlit.app/)

</div>

---

## 1. Project Overview

> **MonsoonCast is an independent machine learning project exploring extreme monsoon rainfall prediction and probabilistic uncertainty estimation over India. It investigates how historical rainfall observations and atmospheric variables can be processed and used with machine learning models and ensemble techniques to estimate the likelihood of extreme rainfall events.**

The framework focuses on the **Central India Core Monsoon Zone** (18.0°–25.0°N, 73.0°–82.0°E) during the Indian Summer Monsoon season (June–September; JJAS). It implements an end-to-end, leak-free machine learning workflow:
- **Strict Chronological Data Splitting:** Isolates training ($\le 2018$), validation (2019–2020), and test partitions (2021–2023) to eliminate temporal lookahead.
- **Physics-Informed Feature Engineering:** Computes multi-day backward lags, rolling accumulators, 2D mesoscale neighborhood spatial filters, and training-only climatology.
- **Multi-Model Benchmark:** Evaluates climatological priors, persistence, L2 logistic regression, cost-sensitive Random Forests, gradient-boosted decision trees (XGBoost), and deep neural networks.
- **Constrained Probabilistic Ensembling:** Optimizes non-negative member weights on held-out validation data and applies post-hoc isotonic calibration.
- **Uncertainty Quantification:** Estimates empirical ensemble spread, Shannon disagreement entropy, and continuous conditional precipitation prediction intervals via quantile regression.
- **Dependence-Aware Evaluation:** Implements a **Spatial-Date Block Bootstrap** (5-day synoptic date clusters) to estimate 95% confidence intervals while preserving spatial and temporal correlation.

---

## 2. Motivation

The Indian Summer Monsoon supplies over 75% of India's annual freshwater replenishment, driving the hydrology, agriculture, and economy of South Asia. However, monsoon precipitation is characterized by severe spatio-temporal variability. Mesoscale convective complexes and synoptic depressions frequently trigger localized extreme downpours ($\ge 50$ mm/day) that cause catastrophic flash floods, landslides, and infrastructure damage.

From a machine learning engineering perspective, extreme monsoon rainfall presents several compelling technical challenges:
1. **Extreme Class Imbalance:** Extreme events occur on less than 2% of grid-day instances across the monsoon season, making standard loss functions and naive accuracy metrics misleading.
2. **Spatio-Temporal Autocorrelation:** Weather systems persist across days and span contiguous geographic regions. Standard randomized cross-validation causes severe temporal leakage, producing artificially optimistic evaluations.
3. **Probability Miscalibration:** Modern high-capacity models (such as gradient-boosted trees and deep networks) frequently exhibit extreme overconfidence for rare positive events, predicting inflated probabilities that do not reflect observed event frequencies.
4. **Predictive Uncertainty:** Deterministic point forecasts fail to convey operational confidence. Quantifying inter-model spread and continuous prediction intervals provides critical context for risk assessment.

MonsoonCast was developed to explore these machine learning challenges through a principled, reproducible experimental testbed.

---

## 3. Problem Statement

Given historical meteorological observations up to day $t$ across a regular spatial grid over Central India, the objective is to:

1. **Binary Extreme Event Forecasting:** Estimate the calibrated probability that 24-hour rainfall $R$ at grid coordinate $s$ on the following day $t+1$ exceeds the operational heavy rainfall threshold:
   $$Y(s, t+1) = \mathbb{I}(R(s, t+1) \ge \tau), \quad \text{where } \tau = 50.0\text{ mm/day}$$

2. **Continuous Prediction Intervals:** Estimate the conditional rainfall quantiles $Q_\alpha(s, t+1)$ for $\alpha \in \{0.10, 0.50, 0.90\}$ to bound next-day precipitation amounts:
   $$\mathbb{P}\left( Q_{0.10}(s, t+1) \le R(s, t+1) \le Q_{0.90}(s, t+1) \right) \approx 0.80$$

3. **Strict Causal Operational Constraint:** All predictors used to forecast conditions on day $t+1$ must be strictly available at forecast issue time $t$. No future observations or uninitialised target-period variables may enter the feature matrix.

---

## 4. Project Objectives

- **Zero-Leakage Pipeline:** Build a robust data preparation pipeline enforcing strict calendar alignment, continuous multi-day seasonal buffers, and chronological year-based partitioning.
- **Diverse Model Architectures:** Implement and systematically compare baseline estimators, regularized linear models, tree-based ensembles, neural networks, and quantile regressors.
- **Probabilistic Calibration:** Integrate post-hoc calibration techniques (Platt scaling, Isotonic regression) fitted exclusively on validation predictions to eliminate probability distortion.
- **Ensemble Optimization:** Formulate a constrained quadratic program to determine optimal non-negative ensemble weights based on validation performance.
- **Comprehensive Verification:** Measure classification discrimination (ROC-AUC, PR-AUC), rare-event contingency scores (POD, FAR, CSI), and probabilistic calibration (Brier Skill Score, ECE) with spatial-date block-bootstrap confidence intervals.
- **Modular Software Architecture:** Package the framework as an installable Python package supported by a FastAPI inference microservice, an interactive Streamlit analytical dashboard, and an automated test suite.

---

## 5. Data Sources

### 5.1 Primary Target Data (IMD Gridded Daily Rainfall)
- **Source:** India Meteorological Department (IMD) National Climate Center.
- **Format:** Gridded NetCDF files (`rain` variable, units: mm/day).
- **Spatial Resolution:** 0.25° regular latitude-longitude grid (~27 km).
- **Temporal Coverage:** Daily 24-hour accumulated rainfall (08:30 IST to 08:30 IST).
- **Masking:** Ocean cells and unmonitored border areas are coded with missing value `-999.0` and excluded.
- **Domain:** Central India Core Monsoon Zone (18.0°–25.0°N, 73.0°–82.0°E).

### 5.2 Atmospheric Reanalysis (ERA5 Mode Extension)
- **Source:** European Centre for Medium-Range Weather Forecasts (ECMWF) Copernicus Climate Change Service (C3S).
- **Variables:** Mean Sea Level Pressure (`msl`, hPa), Total Column Water Vapour (`tcwv`, kg/m²), 850 hPa Zonal Wind (`u850`, m/s), 850 hPa Meridional Wind (`v850`, m/s), 500 hPa Geopotential Height (`z500`, gpm).
- **Operational Classification:** *Retrospective Diagnostic Association.* Because reanalysis assimilation incorporates observations after the fact, these variables serve to explore physical atmospheric relationships offline, rather than an operational real-time forecast.

### 5.3 Deterministic Synthetic Test Fixture
- To enable immediate local development, automated testing, and CI/CD without requiring multi-gigabyte external downloads, MonsoonCast includes a synthetic generator (`generate_synthetic_monsoon_dataset`) producing physically patterned monsoon rain fields on a 0.50° grid.
- **Data Governance Policy:** Synthetic fixtures are used strictly for software pipeline verification, causal testing, and integration. Results generated on the synthetic fixture are explicitly documented in `artifacts/experiments/provenance_manifest.json` and must not be cited as empirical atmospheric measurements.

---

## 6. System Architecture

```
monsooncast/
├── configs/                # Validated YAML experiment configurations
│   ├── data.yaml           # Spatial bounding (Central India) & temporal scopes
│   ├── baseline.yaml       # Baselines, tree models, and neural architecture
│   ├── ensemble.yaml       # Ensemble weights and calibration configuration
│   └── reanalysis.yaml     # Coupled atmospheric reanalysis experiment
├── data/
│   ├── raw/                # Source IMD / ERA5 NetCDF archives or synthetic fixtures
│   ├── interim/            # Coordinate-aligned spatial subsets
│   └── processed/          # Parquet feature tables with aligned next-day targets
├── docs/                   # Technical architecture, methodology, and data dictionary
├── src/monsooncast/        # Core Python package
│   ├── config.py           # Configuration parsing & SHA-256 experiment hashing
│   ├── data/               # NetCDF ingestion, missing-value audits, synthetic generators
│   ├── features/           # Backward lags, rolling sums, spatial 3x3 filters, climatology
│   ├── targets/            # Extreme event thresholding and multi-quantile definitions
│   ├── models/             # Climatology, Persistence, LogReg, RF, XGBoost, PyTorch MLP
│   ├── calibration/        # Platt scaling and non-parametric isotonic calibrators
│   ├── evaluation/         # POD, FAR, CSI, BSS, ECE, and Spatial-Date Block Bootstrap
│   └── visualization/      # Plotly and Matplotlib publication figures
├── scripts/                # CLI executable pipeline scripts
│   ├── inspect_data.py     # Meteorological NetCDF inspection and coordinate validation
│   ├── prepare_data.py     # Feature engineering and chronological partitioning
│   ├── train.py            # Model fitting, ensemble weighting, and artifact serialization
│   ├── evaluate.py         # Held-out testing and block-bootstrap CI calculation
│   ├── analyze_feature_importance.py # Permutation importance and XGBoost gain
│   ├── sensitivity_analysis.py       # Threshold and sub-regional sensitivity
│   ├── generate_figures.py # Automated generation of publication-quality PNG figures
│   └── run_experiment.py   # Master end-to-end pipeline runner
├── api/                    # FastAPI REST API with Pydantic request/response schemas
├── app/                    # 3-page interactive Streamlit dashboard
├── artifacts/              # Serialized models, metrics, and provenance manifest
└── tests/                  # Pytest automated test suite (27 passing tests)
```

---

## 7. Data Processing Pipeline

```
Raw NetCDF (IMD / Synthetic) 
      │
      ▼
Spatial Subsetting (18°–25°N, 73°–82°E)
      │
      ▼
Continuous Temporal Buffering (May 15 – Oct 15)
      │
      ▼
Feature Engineering (Lags, Rolling Stats, 3x3 Spatial Filters, DOY Climatology)
      │
      ▼
Target Lead Shift (Shift -1 within year block; target valid strictly in JJAS)
      │
      ▼
Strict Chronological Partitioning:
├── Train Set (2010–2018): 310,365 samples (Used for model training & climatology)
├── Val Set   (2019–2020):  68,970 samples (Used for ensemble weights & calibration)
└── Test Set  (2021–2023): 103,455 samples (Strictly held out for final evaluation)
```

1. **Spatial Subsetting:** Restricts latitude to [18.0°, 25.0°] and longitude to [73.0°, 82.0°], isolating the core monsoon zone. Ocean grid points are masked out.
2. **Seasonal Continuity Buffers:** Time series start on May 15 and end on October 15 for each year. This guarantees that June 1 has valid antecedent lags (from late May) and September 30 targets can be evaluated against October 1 observations.
3. **Year-Bounded Target Alignment:** Lead shifting is grouped strictly by calendar year (`df.groupby('year')`), preventing the end of one season (September 30) from bleeding into the beginning of the next (June 1).

---

## 8. Feature Engineering

At simulated issue date $t$, the feature pipeline generates 20 meteorological predictors using strictly antecedent observations:

| Feature Name | Type | Physical Rationale |
| :--- | :--- | :--- |
| `rain_current` | Point observation | Precipitation accumulation on forecast issue day $t$. |
| `rain_lag1`, `rain_lag2`, `rain_lag3`, `rain_lag7` | Backward lags | Captures short-term memory (1–3 days) and weekly synoptic persistence. |
| `rain_roll_sum3`, `rain_roll_sum7` | Rolling sums | Cumulative soil moisture preconditioning and convective loading. |
| `rain_roll_max7`, `rain_roll_mean7` | Rolling extrema | Historical storm intensity over the preceding week. |
| `rain_spatial_mean3x3` | Mesoscale spatial filter | 3x3 grid average capturing surrounding convective organization. |
| `rain_spatial_max3x3` | Mesoscale spatial filter | Peak rainfall in neighboring cells, detecting organized mesoscale systems. |
| `climatology_mean` | Historical baseline | Grid-cell DOY mean computed strictly from training seasons ($\le 2018$). |
| `doy_sin`, `doy_cos` | Cyclical calendar | Harmonic representations of the annual cycle, isolating active/break phases. |
| `era5_tcwv` *(optional)* | Thermodynamic state | Total Column Water Vapour (atmospheric moisture column loading). |
| `era5_msl` *(optional)* | Dynamic state | Mean Sea Level Pressure (identifying monsoon trough depressions). |
| `era5_u850`, `era5_v850`, `era5_wind_speed850` *(optional)* | Kinematic state | Low-level monsoon westerly jet speed and direction. |
| `era5_z500` *(optional)* | Synoptic state | 500 hPa geopotential height (mid-tropospheric ridges and troughs). |

---

## 9. Machine Learning Models

1. **Climatology Baseline:** Estimates the empirical probability of extreme precipitation for each grid cell and day of year based on training seasons.
2. **Persistence Baseline:** Evaluates next-day probability conditioned on the current day's binary extreme status:
   $$\mathbb{P}(Y_{t+1}=1 \mid Y_t=1) \quad \text{vs.} \quad \mathbb{P}(Y_{t+1}=1 \mid Y_t=0)$$
3. **Penalized Logistic Regression:** L2-regularized generalized linear model with balanced class weighting to handle event rarity.
4. **Random Forest Classifier:** Ensemble of 100 decision trees (max depth 8) with balanced class weights to penalize minority-class false negatives.
5. **XGBoost Classifier:** Gradient-boosted trees optimized with `scale_pos_weight=5.0` to balance recall and precision on rare extremes.
6. **PyTorch Tabular MLP:** Neural network architecture with batch normalization, ReLU activations, and Monte Carlo Dropout for epistemic uncertainty sampling.
7. **XGBoost Quantile Regressors:** Independent gradient-boosted trees trained under the asymmetric pinball loss at $\alpha \in \{0.10, 0.50, 0.90\}$ to output continuous prediction intervals.

---

## 10. Ensemble Modelling

MonsoonCast combines individual model probabilities using a **validation-optimized probabilistic ensemble**. Rather than simple averaging, non-negative weights are fitted by minimizing mean squared error on the validation partition:

$$\min_{\mathbf{w}} \sum_{i=1}^{N_{\text{val}}} \left( y_i - \sum_{k=1}^K w_k \hat{p}_{ik} \right)^2 \quad \text{subject to} \quad \sum_{k=1}^K w_k = 1, \quad w_k \ge 0$$

This formulation leverages the complementary strengths of diverse model families (linear boundary, tree bagging, gradient boosting) while preventing overfitting on the training set.

---

## 11. Probability Calibration and Uncertainty Estimation

### 11.1 Isotonic Probability Calibration
Raw model outputs from decision trees and neural networks frequently suffer from poor calibration on rare events. We fit non-parametric **Isotonic Regression** strictly on validation set predictions:
$$\min_{m} \sum_{i=1}^{N_{\text{val}}} \left( y_i - m(\hat{p}_i) \right)^2 \quad \text{subject to } m(a) \le m(b) \text{ whenever } a \le b$$
Post-hoc isotonic calibration reduces Expected Calibration Error (ECE) from $> 0.30$ to **0.001**, ensuring predicted probabilities match observed event frequencies.

### 11.2 Uncertainty Quantification
- **Empirical Ensemble Spread ($\sigma_{\text{ens}}$):** Standard deviation across ensemble member probabilities:
  $$\sigma_{\text{ens}} = \sqrt{\frac{1}{K}\sum_{k=1}^K (p_k - \bar{p})^2}$$
  Serves as an empirical metric of model disagreement, signaling elevated sensitivity during chaotic convective situations.
- **Consensus Entropy ($H$):** Shannon entropy of the calibrated consensus forecast:
  $$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$
- **Quantile Prediction Intervals:** Evaluated via Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW) between the 10th and 90th percentiles.

---

## 12. Evaluation Metrics

Model performance is evaluated across three complementary verification dimensions:

1. **Rare-Event Contingency Metrics:**
   - $\text{POD} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ (Probability of Detection / Recall)
   - $\text{FAR} = \frac{\text{FP}}{\text{TP} + \text{FP}}$ (False Alarm Ratio)
   - $\text{CSI} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ (Critical Success Index / Threat Score)
2. **Probabilistic & Discrimination Metrics:**
   - Precision-Recall AUC (PR-AUC) and ROC-AUC
   - Brier Score (BS) and Brier Skill Score relative to climatology ($\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{clim}}}$)
   - Expected Calibration Error (ECE)
3. **Spatial-Date Block Bootstrap:**
   To account for spatial and temporal autocorrelation across monsoon storms, confidence intervals are computed by resampling 5-day continuous synoptic date blocks with replacement ($B = 500$). All 285 spatial grid cells for each sampled date are kept together, preserving physical cross-sectional dependence.

---

## 13. Installation and Setup

### Prerequisites
- Python 3.11 or 3.12
- Git

### Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/khushaalsrivastava444-cmyk/MonsoonCast.git
cd MonsoonCast

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies and the monsooncast package in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 14. Project Structure

```
MonsoonCast/
├── configs/                # Configuration specifications (YAML)
│   ├── data.yaml           # Spatial bounds, temporal splits, target threshold
│   ├── baseline.yaml       # Baselines, tree models, and neural network parameters
│   ├── ensemble.yaml       # Weighting schemes and calibration methods
│   └── reanalysis.yaml     # Coupled reanalysis experiment setup
├── data/
│   ├── raw/                # Source meteorological data and test fixtures
│   ├── interim/            # Coordinate-standardized intermediate arrays
│   └── processed/          # Parquet feature datasets (train, val, test)
├── docs/                   # Methodological and architectural documentation
│   ├── architecture.md     # In-depth system design & dataflow
│   ├── data_dictionary.md  # Detailed feature specifications & units
│   ├── methodology.md      # Mathematical formulation of models & calibration
│   └── limitations.md      # Known constraints & scientific boundaries
├── src/monsooncast/        # Core package source code
│   ├── config.py           # Configuration parsing & validation
│   ├── data/               # NetCDF loaders and synthetic generators
│   ├── features/           # Lagging, rolling statistics, spatial filtering
│   ├── targets/            # Target creation & lead-time shifting
│   ├── models/             # Implementations of baselines, trees, and neural nets
│   ├── calibration/        # Platt scaling & Isotonic regression
│   ├── evaluation/         # Contingency metrics, reliability, & block bootstrap
│   └── visualization/      # Spatial mapping and diagnostic plotting
├── scripts/                # Command-line execution tools
│   ├── inspect_data.py     # Check dataset health and spatial bounds
│   ├── prepare_data.py     # Generate feature tables and chronological splits
│   ├── train.py            # Train models and calibrate ensemble
│   ├── evaluate.py         # Calculate benchmark scores & bootstrap CIs
│   ├── analyze_feature_importance.py # Permutation feature rankings
│   ├── sensitivity_analysis.py       # Multi-threshold & regional evaluation
│   ├── generate_figures.py # Create 7 publication-quality PNG figures
│   └── run_experiment.py   # Master end-to-end pipeline runner
├── api/                    # Programmatic serving via FastAPI
│   └── main.py             # REST endpoints for event probabilities & quantiles
├── app/                    # Web-based analytical dashboard via Streamlit
│   ├── streamlit_app.py    # Main landing page & pipeline health monitor
│   └── pages/
│       ├── 1_Rainfall_Explorer.py     # Spatial map & climatology browser
│       ├── 2_Model_Comparison.py      # Verification tables & sensitivity tabs
│       └── 3_Uncertainty_Analysis.py  # Reliability curves & ensemble spread
├── artifacts/              # Generated outputs (models, metrics, manifest)
│   ├── models/             # Serialized joblib model weights
│   ├── metrics/            # Evaluation results & reliability curves (JSON)
│   └── experiments/        # Formal provenance manifest
├── reports/
│   ├── figures/            # 7 generated scientific publication figures (PNG)
│   └── research_report.md  # Detailed benchmark report
└── tests/                  # Automated unit, integration, and causal test suite
```

---

## 15. Usage Instructions

### 15.1 Running the End-to-End Pipeline

```bash
# Step 1: Inspect raw data or generate test fixture
python scripts/inspect_data.py

# Step 2: Extract features and create chronological splits
python scripts/prepare_data.py --config configs/data.yaml --include-era5

# Step 3: Train all models and calibrate ensemble
python scripts/train.py --config configs/reanalysis.yaml

# Step 4: Evaluate on held-out test data with Spatial-Date Block Bootstrap CIs
python scripts/evaluate.py

# Step 5: Compute permutation feature importance
python scripts/analyze_feature_importance.py

# Step 6: Run multi-threshold and sub-regional sensitivity analysis
python scripts/sensitivity_analysis.py

# Step 7: Generate scientific diagnostic figures in reports/figures/
python scripts/generate_figures.py
```

Alternatively, execute the complete pipeline in a single command:
```bash
python scripts/run_experiment.py --config configs/reanalysis.yaml
```

### 15.2 Interactive Streamlit Dashboard

- **Live Cloud Demo:** [monsooncast-popwidywnbpu3rgwkn5skh.streamlit.app](https://monsooncast-popwidywnbpu3rgwkn5skh.streamlit.app/)
- **Run locally:**
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` to explore:
- **Rainfall Explorer:** Spatio-temporal rainfall maps with OpenStreetMap and clean vector basemaps (no API key required).
- **Model Comparison:** Interactive metric tables, ROC/PR curves, predictor importance, and sensitivity analyses.
- **Uncertainty Analysis:** Reliability diagrams, Expected Calibration Error (ECE), and quantile prediction intervals.

### 15.3 Programmatic FastAPI Service

- **Live Cloud Endpoint:** [monsooncast-api.onrender.com/docs](https://monsooncast-api.onrender.com/docs)
- **Run locally:**
```bash
uvicorn api.main:app --reload --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs` (or `/docs` on the live hosted URL).

**Example Prediction Request:**
```bash
curl -X POST "http://localhost:8000/predict/event-probability" \
     -H "Content-Type: application/json" \
     -d '{
       "model_name": "ensemble",
       "samples": [{
         "lat": 21.0,
         "lon": 77.0,
         "doy": 205,
         "rain_current": 42.5,
         "rain_lag1": 31.0,
         "rain_lag2": 15.2,
         "rain_lag3": 5.0,
         "rain_lag7": 0.0,
         "rain_roll_sum3": 88.7,
         "rain_roll_sum7": 112.4,
         "rain_roll_max7": 42.5,
         "rain_roll_mean7": 16.05,
         "rain_spatial_mean3x3": 38.2,
         "rain_spatial_max3x3": 55.0,
         "climatology_mean": 12.8
       }]
     }'
```

### 15.4 Running the Test Suite

```bash
pytest tests/ -v
ruff check src/ scripts/ tests/ api/ app/
```

---

## 16. Current Development Status

> [!IMPORTANT]
> **Data Provenance & Validation Status:**  
> The quantitative metrics presented below represent a **Provisional Benchmark on the Synthetic Validation Fixture (0.50° Grid)** ($N = 103,455$ grid-day instances across held-out test seasons 2021–2023; extreme event prevalence = $1.75\%$).  
> These numbers rigorously verify software pipeline integrity, causal time alignment (no seasonal boundary bleeding), and probability calibration mechanics. They must **not** be cited as empirical atmospheric observations until rerun against archived IMD 0.25° NetCDF files. Provenance details are formalized in `artifacts/experiments/provenance_manifest.json`.

Evaluated on held-out test seasons (2021–2023; $N = 103,455$ grid-day instances: 285 grid cells $\times$ 121 valid issue days per season $\times$ 3 test seasons; extreme event frequency = $1.75\%$). Operational decision thresholds are tuned strictly on the validation partition to eliminate data leakage. All confidence intervals are computed via **Spatial-Date Block Bootstrap** (5-day synoptic date clusters):

| Model Architecture | PR-AUC | ROC-AUC | POD (Recall) | FAR (False Alarm) | CSI [95% CI] | BSS (vs Clim) | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Climatology** | 0.0481 | 0.6142 | 0.121 | 0.914 | 0.053 [0.04–0.06] | +0.000 (Ref) | 0.023 |
| **Lag-1 Persistence** | 0.0592 | 0.5170 | 0.051 | 0.949 | 0.026 [0.02–0.03] | +0.067 | 0.000 |
| **Logistic Regression** | 0.0679 | 0.7866 | 0.263 | 0.911 | 0.071 [0.06–0.08] | -9.339 | 0.360 |
| **Random Forest** | 0.0638 | 0.7815 | 0.309 | 0.913 | 0.073 [0.07–0.08] | -8.115 | 0.334 |
| **XGBoost Classifier** | 0.0667 | 0.7833 | 0.244 | 0.909 | 0.071 [0.06–0.08] | -0.259 | 0.057 |
| **PyTorch Tabular MLP** | 0.0667 | 0.7870 | 0.252 | 0.910 | 0.071 [0.06–0.08] | -9.724 | 0.359 |
| **Calibrated Ensemble** | **0.0673** | **0.7817** | **0.361** | **0.919** | **0.071 [0.07–0.08]** | **+0.088** | **0.001** |

### Key Quantile Interval Results:
- **Nominal Interval:** 80% (10th to 90th percentile)
- **Empirical Coverage (PICP):** **79.4%**
- **Mean Prediction Interval Width (MPIW):** **24.8 mm**

---

## 17. Limitations and Future Improvements

### Technical & Methodological Limitations
1. **Spatial Resolution Limits:** A 0.25° or 0.50° grid averages rainfall over roughly 25–50 km cells. It cannot resolve highly localized convective cloudbursts (< 5 km) or micro-scale terrain interactions.
2. **Land-Only Coverage:** IMD observational products cover terrestrial India. Oceanic convective genesis over the Bay of Bengal and Arabian Sea remains unobserved in rainfall-only models.
3. **Severe Class Imbalance:** Tail events ($\ge 50$ mm/day) constitute $< 2\%$ of samples, limiting the statistical power of standard classification thresholds and requiring careful metric reporting.
4. **Diagnostic vs. Real-Time Availability:** Coupled ERA5 atmospheric variables provide valuable physical diagnostic insights but are not operational real-time forecast inputs due to reanalysis data assimilation latency.

### Planned Enhancements
- **Operational NWP Ingestion:** Ingest archived numerical weather prediction ensemble forecasts (IMD GFS / ECMWF TIGGE) initialized prior to issue time to explore true dynamical-statistical post-processing.
- **Remote Sensing Integration:** Ingest Doppler Weather Radar (DWR) reflectivity and INSAT-3D/3DR geostationary infrared brightness temperatures to capture real-time convective cloud development.
- **Graph Neural Networks (GNNs):** Model spatial atmospheric teleconnections across Central India using spatio-temporal graph convolutional networks.

---

## Citation & Attribution

If you find MonsoonCast useful in your research or personal projects, please cite:

```bibtex
@software{srivastava2026monsooncast,
  author       = {Khushaal Srivastava},
  title        = {{MonsoonCast: An Independent Machine Learning Framework for Extreme Monsoon Rainfall Prediction and Uncertainty Estimation over India}},
  month        = feb,
  year         = 2026,
  version      = {0.3.0},
  url          = {https://github.com/khushaalsrivastava444-cmyk/MonsoonCast}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
