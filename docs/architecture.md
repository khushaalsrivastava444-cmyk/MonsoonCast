# MonsoonCast System Architecture

## 1. Architectural Overview

MonsoonCast is an end-to-end scientific machine learning framework designed for predicting extreme Indian summer monsoon rainfall events and quantifying predictive uncertainty. The system enforces strict chronological validation, modular component decoupling, and transparent data provenance.

```
                           +----------------------------------------+
                           |       Meteorological Data Sources      |
                           |  - IMD 0.25 Gridded Daily NetCDF       |
                           |  - ERA5 Atmospheric Reanalysis         |
                           |  - TIGGE / Numerical Ensemble Archive  |
                           +----------------------------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |        Data Ingestion & Quality        |
                           |  (monsooncast.data.loaders/validation) |
                           |  - Coordinate normalization            |
                           |  - Missing-value audit (-999 flag)     |
                           |  - Spatial & Seasonal bounding (JJAS)  |
                           +----------------------------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |    Feature & Target Construction       |
                           | (monsooncast.features/targets)         |
                           |  - Strictly backward-shifted lags (t-k)|
                           |  - Backward rolling windows (3d, 7d)   |
                           |  - Climatology (Train-period only)     |
                           |  - Extreme Event Label: Y_{t+1} >= tau |
                           +----------------------------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |    Temporal Partitioning (No Leakage)  |
                           |  Train: 2010-2018 | Val: 2019-2020     |
                           |  Test: 2021-2023 (Held-out)            |
                           +----------------------------------------+
                                               |
                                               v
       +-------------------------------------------------------------------------------+
       |                               Model Registry                                  |
       |  Baselines: Climatology, Persistence, Logistic Reg, Ridge                     |
       |  ML: Random Forest, XGBoost Classifier, XGBoost Quantile Regressor            |
       |  Deep Learning: PyTorch Tabular MLP / Temporal NN (with MC Dropout)           |
       +-------------------------------------------------------------------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |    Uncertainty & Calibration Engine    |
                           |  (monsooncast.calibration/ensemble)    |
                           |  - Platt Scaling & Isotonic Regression |
                           |  - Ensemble Probability Averaging      |
                           |  - Validation-loss optimal weights     |
                           |  - Ensemble Spread & Disagreement      |
                           +----------------------------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |         Evaluation & Metrics           |
                           |  - Brier Score & Brier Skill Score     |
                           |  - Reliability Diagrams & ECE          |
                           |  - Contingency: POD, FAR, CSI, ETS     |
                           |  - Block-bootstrapped confidence intervals|
                           +----------------------------------------+
                                               |
                       +-----------------------+-----------------------+
                       |                                               |
                       v                                               v
        +-----------------------------+                 +-----------------------------+
        |   Streamlit Research App    |                 |      FastAPI REST Service   |
        |  - Rainfall Explorer        |                 |  - /health                  |
        |  - Model Comparison         |                 |  - /experiments             |
        |  - Uncertainty Analysis     |                 |  - /predict/event-prob      |
        +-----------------------------+                 +-----------------------------+
```

## 2. Core Modules

1. **`monsooncast.config`**: Pydantic v2 schemas providing type safety and serialization for all data, model, feature, and experiment specifications.
2. **`monsooncast.data`**: Loaders and quality validators for NetCDF/xarray scientific arrays. Handles missing values, bounds checking, unit verification, and realistic synthetic generation.
3. **`monsooncast.features`**: Physics-informed feature generation. Ensures causality: features computed at forecast issue time $t$ predict target at $t+1$. Rolling windows are explicitly shifted by 1 day.
4. **`monsooncast.targets`**: Binary event definitions based on IMD meteorological thresholds (e.g., 50 mm/day heavy rain, 35.5 mm/day moderate-heavy, 64.5 mm/day very heavy, and local 95th/99th percentiles).
5. **`monsooncast.models`**: Modular model implementations including climatological priors, persistence, penalized generalized linear models, gradient-boosted trees, and neural networks.
6. **`monsooncast.calibration`**: Probability calibration tools (Platt scaling and isotonic regression) fitted strictly on validation partitions to prevent overconfident probabilistic forecasts.
7. **`monsooncast.evaluation`**: Comprehensive meteorological verification scores (Critical Success Index, False Alarm Ratio, Probability of Detection, Brier Skill Score) and reliability curves.
8. **`api` & `app`**: Production-grade microservices for RESTful programmatic access and an interactive Streamlit scientific dashboard.

## 3. Data Flow & Leakage Guardrails

- **Zero Temporal Leakage:** Data are partitioned chronologically by calendar year. All feature standardizers, imputers, rolling climatological statistics, and calibration curves are fitted exclusively on the training period.
- **Zero Spatial Leakage:** Central India test evaluations evaluate regional spatial coherence without mixing future spatial states.
- **Traceability:** Every model artifact produced includes its config hash, parameter inventory, and training metadata.
