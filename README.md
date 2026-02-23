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

### Run All Steps in One Command:
```bash
python scripts/run_experiment.py --config configs/baseline.yaml
```

---

## 5. Interactive Research Dashboard

Launch the Streamlit analytical dashboard:
```bash
streamlit run app/streamlit_app.py
```

### Pages Included:
- **1. Rainfall Explorer:** Spatio-temporal rainfall distributions, seasonal progression, and interactive density map of Central India.
- **2. Model Comparison:** Side-by-side verification table (PR-AUC, ROC-AUC, POD, FAR, CSI, BSS, ECE).
- **3. Uncertainty Analysis:** Reliability curves, expected calibration error (ECE), and quantile interval coverage (PICP).

---

## 6. Programmatic REST API

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

## 7. Automated Testing Suite

Execute the full unit, integration, and leakage test suite:
```bash
pytest tests/ -v
```

---

## 8. Citation & Attribution

If you use MonsoonCast in your academic research, please cite:

```bibtex
@software{srivastava2026monsooncast,
  author       = {Khushaal Srivastava},
  title        = {{MonsoonCast: A Research-Grade AI Ensemble Framework for Extreme Monsoon Rainfall Prediction and Uncertainty Quantification over India}},
  month        = feb,
  year         = 2026,
  version      = {0.1.0},
  url          = {https://github.com/khushaalsrivastava/MonsoonCast}
}
```

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
