<div align="center">

# MonsoonCast 🌧️
### A Research-Grade AI Ensemble Framework for Extreme Monsoon Rainfall Prediction and Uncertainty Quantification over India

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release Date](https://img.shields.io/badge/release-2026--02--23-orange.svg)](CITATION.cff)
[![CI Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](.github/workflows/tests.yml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](http://localhost:8000/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](http://localhost:8501)

*Developed for Advanced Climate AI Research, Undergraduate Thesis Portfolio, and IIT Bombay Research Applications.*

</div>

---

## 1. Executive Summary & Research Vision

The Indian Summer Monsoon (June–September; JJAS) delivers over 75% of India's annual precipitation. Extreme rainfall events ($\ge 50$ mm/day) frequently trigger catastrophic flash floods, landslides, and agricultural disruption. However, accurately forecasting extreme localized precipitation remains one of the grand challenges in atmospheric science due to severe class imbalance, mesoscale convective organization, and rapid chaotic error growth.

**MonsoonCast** addresses this challenge through a modular, scientifically grounded AI framework that:
1. **Prevents Future Leakage:** Enforces strict chronological year-based partitioning (Train $\le 2018$, Val 2019–2020, Test 2021–2023) and strictly backward-shifted rolling windows.
2. **Combines Diverse Priors:** Evaluates seasonal climatology, persistence, regularized GLMs, Random Forests, XGBoost, and deep neural networks.
3. **Optimizes Calibration:** Fits non-negative ensemble weights and probability calibrators (Platt scaling and Isotonic regression) strictly on validation data to mitigate extreme-event overconfidence.
4. **Quantifies Epistemic & Aleatoric Uncertainty:** Provides ensemble member spread, disagreement entropy, and conditional rainfall prediction intervals via gradient-boosted quantile regression.

---

## 2. System Architecture

```
monsooncast/
├── configs/                # Pydantic-validated YAML experiment configurations
│   ├── data.yaml           # Spatial bounding (Central India) & temporal scopes
│   ├── baseline.yaml       # Baselines and ML models
│   ├── ensemble.yaml       # Probabilistic weighting and calibration
│   └── forecast.yaml       # NWP forecast-informed extension
├── data/
│   ├── raw/                # Immutable IMD / ERA5 source NetCDFs
│   ├── interim/            # Coordinate-aligned, seasonally filtered arrays
│   └── processed/          # Parquet feature tensors with aligned targets
├── docs/                   # Scientific methodology, architecture, and limitations
├── src/monsooncast/
│   ├── config.py           # Pydantic schemas and SHA-256 config hashing
│   ├── data/               # NetCDF loaders, missing audits, coordinate checks
│   ├── features/           # Backward lags, rolling sums, spatial 3x3 filters
│   ├── targets/            # Extreme rainfall labels (>= 50 mm/day, percentiles)
│   ├── models/             # Climatology, Persistence, LogReg, RF, XGBoost, PyTorch
│   ├── calibration/        # Platt scaling & Isotonic probability calibration
│   ├── evaluation/         # POD, FAR, CSI, ETS, HSS, Brier Skill Score, CI
│   └── visualization/      # Matplotlib and Plotly spatial maps
├── scripts/
│   ├── inspect_data.py     # Meteorological array inspection & validation
│   ├── prepare_data.py     # Feature engineering & chronological splitting
│   ├── train.py            # Model training & artifact serialization
│   ├── evaluate.py         # Held-out testing & bootstrap confidence intervals
│   └── run_experiment.py   # End-to-end master experiment execution
├── api/                    # FastAPI REST microservice with typed Pydantic schemas
├── app/                    # 3-page Streamlit scientific research dashboard
└── tests/                  # Pytest unit, integration, and leakage test suite
```

---

## 3. Quickstart & Installation

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/khushaalsrivastava/MonsoonCast.git
cd MonsoonCast

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

---

## 4. Reproducing Experiments

### Step 1: Inspect Dataset
Inspect real IMD NetCDF files or generate a synthetic test fixture:
```bash
python scripts/inspect_data.py
```

### Step 2: Prepare Features (Zero Leakage)
Extract lagged predictors, rolling statistics, and chronological partitions:
```bash
python scripts/prepare_data.py --config configs/data.yaml
```

### Step 3: Train Models & Calibrated Ensemble
```bash
python scripts/train.py --config configs/baseline.yaml
```

### Step 4: Evaluate with Block-Bootstrap Confidence Intervals
```bash
python scripts/evaluate.py
```

### Step 5: Meteorological Interpretability & Sensitivity Analysis
```bash
python scripts/analyze_feature_importance.py
python scripts/sensitivity_analysis.py
```

### Step 6: Generate Publication-Quality Scientific Figures
```bash
python scripts/generate_figures.py
```

### Run End-to-End Experiment Pipeline:
```bash
python scripts/run_experiment.py --config configs/reanalysis.yaml
```

---

## 5. Quantitative Benchmark Results

Evaluated on held-out test seasons (2021–2023; $N = 104,025$ grid-day instances; extreme event frequency = $1.74\%$). Operational decision threshold tuned strictly on the validation partition to eliminate data leakage:

| Model Architecture | PR-AUC | ROC-AUC | POD (Recall) | FAR (False Alarm) | CSI [95% CI] | BSS (vs Clim) | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Climatology** | 0.0457 | 0.6144 | 0.344 | 0.952 | 0.044 [0.04–0.05] | +0.000 (Ref) | 0.022 |
| **Lag-1 Persistence** | 0.0673 | 0.5212 | 0.059 | 0.941 | 0.030 [0.02–0.04] | +0.067 | 0.001 |
| **Logistic Regression** | 0.0654 | 0.7950 | 0.311 | 0.920 | 0.068 [0.06–0.07] | -9.402 | 0.360 |
| **Random Forest** | 0.0616 | 0.7860 | 0.327 | 0.924 | 0.066 [0.06–0.07] | -8.125 | 0.333 |
| **XGBoost Classifier** | 0.0646 | 0.7913 | 0.356 | 0.924 | 0.067 [0.06–0.07] | -0.259 | 0.056 |
| **PyTorch Tabular MLP** | 0.0656 | 0.7948 | 0.300 | 0.921 | 0.067 [0.06–0.07] | -9.900 | 0.361 |
| **Calibrated Ensemble** | **0.0648** | **0.7901** | **0.385** | **0.927** | **0.065 [0.06–0.07]** | **+0.088** | **0.001** |

---

## 6. Publication Figures

Seven publication-quality scientific figures are automatically generated in `reports/figures/`:
1. `fig1_study_domain_climatology.png`: Spatial domain map & intra-seasonal progression.
2. `fig2_model_pr_roc_curves.png`: Precision-Recall & ROC curves.
3. `fig3_reliability_diagrams.png`: Reliability diagrams & Expected Calibration Error (ECE).
4. `fig4_prediction_intervals.png`: Quantile regression conditional prediction intervals.
5. `fig5_spatial_forecast_uncertainty.png`: Calibrated event probability & inter-model spread $\sigma$.
6. `fig6_feature_importance.png`: Permutation feature importance & XGBoost information gain.
7. `fig7_threshold_subregion_sensitivity.png`: Operational threshold response and sub-regional verification.

---

## 7. Interactive Research Dashboard

Launch the Streamlit analytical dashboard:
```bash
streamlit run app/streamlit_app.py
```

### Pages Included:
- **1. Rainfall Explorer:** Spatio-temporal rainfall distributions, seasonal progression, and interactive density map of Central India.
- **2. Model Comparison:** Side-by-side verification table (PR-AUC, ROC-AUC, POD, FAR, CSI, BSS, ECE), plus meteorological predictor importance and sensitivity analysis tabs.
- **3. Uncertainty Analysis:** Reliability curves, expected calibration error (ECE), and quantile interval coverage (PICP).

---

## 8. Programmatic REST API

Launch the FastAPI service:
```bash
uvicorn api.main:app --reload --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### Key Endpoints:
- `GET /health` — Check system and artifact availability.
- `GET /metadata` — Domain boundaries, supported models, and thresholds.
- `GET /experiments` — Registered experiment runs and metrics.
- `POST /predict/event-probability` — Consensus calibrated probability, member spread, and entropy.
- `POST /predict/rainfall-quantiles` — 10th, 50th, 90th conditional rainfall quantiles.

---

## 9. Automated Testing Suite

Execute the full unit, integration, and leakage test suite:
```bash
pytest tests/ -v
```

---

## 10. Citation & Attribution

If you use MonsoonCast in your academic research, please cite:

```bibtex
@software{srivastava2026monsooncast,
  author       = {Khushaal Srivastava},
  title        = {{MonsoonCast: A Research-Grade AI Ensemble Framework for Extreme Monsoon Rainfall Prediction and Uncertainty Quantification over India}},
  month        = feb,
  year         = 2026,
  version      = {0.3.0},
  url          = {https://github.com/khushaalsrivastava/MonsoonCast}
}
```

---

## 11. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
