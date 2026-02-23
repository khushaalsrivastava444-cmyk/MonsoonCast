# Scientific & Engineering Limitations

In adherence to research integrity, this document explicitly details the theoretical, empirical, and computational limitations of the MonsoonCast framework.

## 1. Domain and Observational Constraints
1. **IMD Gridded Spatial Resolution (0.25° ~ 27 km):**
   - High-intensity convective events such as localized cloudbursts often occur on spatial scales of 1–5 km and sub-daily timeframes. A 0.25° gridded daily average smooths out localized peak intensity.
2. **Land-Masking:**
   - IMD gridded products cover land areas of India. Oceanic precipitation over the Arabian Sea and Bay of Bengal (crucial for monsoon depression genesis) is unmeasured in this dataset.
3. **Missing Value Conventions:**
   - Coastal and mountainous grid cells with sparse rain-gauge density may experience intermittent missing values (-999.0).

## 2. Modeling & Methodological Boundaries
1. **Antecedent Rainfall Predictability Limit:**
   - Rainfall history alone (lags, rolling sums) contains strong persistence and seasonal signal, but cannot dynamically predict spontaneous convective triggering without thermodynamic and dynamic atmospheric predictors.
2. **Reanalysis vs Real-Time NWP:**
   - ERA5 reanalysis combines observations and model physics retrospectively. Retrospective diagnostic experiments must never be characterized as operational forecasts.
3. **Severe Class Imbalance:**
   - Extreme events ($\ge 50$ mm/day) typically constitute $< 2\%$ to $5\%$ of daily summer monsoon grid observations. Accuracy is therefore an invalid evaluation metric; threshold selection strongly balances POD against FAR.

## 3. Uncertainty Quantification Scope
1. **Ensemble Spread:**
   - Model-weight or algorithmic ensemble spread captures *epistemic* (model form / parameter) uncertainty, not full *aleatoric* atmospheric chaos.
2. **Calibration Post-Processing:**
   - Isotonic regression and Platt scaling assume monotonic calibration curves and stationary validation distributions. Climate non-stationarity or anomalous monsoon years may induce calibration drift.
