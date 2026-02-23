# MonsoonCast: A Research-Grade AI Ensemble Framework for Extreme Monsoon Rainfall Prediction and Uncertainty Quantification over India

**Author:** Khushaal Srivastava  
**Affiliation:** AI & Climate Data Science Research Initiative  
**Release Date:** February 23, 2026  
**Document Status:** Working Research Paper / Benchmark Report  

---

## 1. Abstract

Extreme precipitation events during the Indian Summer Monsoon (June through September; JJAS) present severe risks to human life, urban infrastructure, and agrarian economies. Numerical Weather Prediction (NWP) models frequently suffer from convective parameterization biases and spatial displacement errors when resolving localized extreme downpours ($\ge 50$ mm/day). In this work, we present **MonsoonCast**, a modular, reproducible, physics-aware machine learning ensemble framework designed for extreme monsoon precipitation forecasting and rigorous predictive uncertainty quantification over Central India (18.0°–25.0°N, 73.0°–82.0°E). 

To ensure scientific validity, the framework enforces strict chronological partitioning (Train: 2010–2018, Validation: 2019–2020, Test: 2021–2023), eliminating forward temporal look-ahead and target leakage. We evaluate seasonal climatology, lag persistence, penalized logistic regression, cost-sensitive Random Forests, and gradient-boosted decision trees (XGBoost) against a constrained, validation-optimized probabilistic ensemble with post-hoc isotonic calibration. Experimental results demonstrate that ensembling substantially outperforms individual estimators on probability discrimination (PR-AUC), Brier Skill Score (BSS) relative to climatology, and Expected Calibration Error (ECE). Furthermore, we quantify epistemic uncertainty using inter-model spread and estimate continuous conditional precipitation prediction intervals via gradient-boosted quantile regression.

---

## 2. Introduction and Motivation

The Indian Summer Monsoon is the primary driver of South Asia's hydrological cycle, supplying more than 75% of India's annual freshwater replenishment. However, the intra-seasonal distribution of monsoon rainfall is characterized by extreme spatio-temporal variability. Mesoscale convective systems, monsoon depressions originating over the Bay of Bengal, and low-level jet interactions with the Western Ghats frequently trigger catastrophic flash flooding and landslides.

Traditional physical weather forecasting systems struggle with extreme precipitation due to:
1. Coarse computational grid spacing (~10–25 km) that cannot explicitly resolve cloud microphysics.
2. Exponential sensitivity to initial boundary condition errors (atmospheric chaos).
3. Calibration drift, wherein raw model probabilities systematically misestimate the occurrence frequency of rare tail events.

Machine learning offers significant promise for post-processing and diagnostic precipitation modeling. However, many published AI climate benchmarks suffer from subtle methodological flaws: random cross-validation on autocorrelated time series (temporal leakage), selective reporting of overall accuracy on severely imbalanced datasets, and lack of rigorous probability calibration. MonsoonCast addresses these deficiencies by establishing a mathematically sound, leak-free experimental testbed.

---

## 3. Research Questions

This study investigates five core research questions:
1. **RQ1 (Predictive Skill):** Can a machine learning ensemble demonstrate statistically significant improvements in Critical Success Index (CSI) and Brier Skill Score (BSS) compared with historical climatological and persistence baselines?
2. **RQ2 (Feature Utility):** Which engineered antecedent features (lags, rolling multi-day sums, spatial 3x3 convective neighborhood statistics) provide the strongest signal for next-day extreme event triggering?
3. **RQ3 (Calibration & Reliability):** Does validation-fitted isotonic regression effectively eliminate the overconfidence typically observed in raw gradient-boosted trees for rare classes ($P(Y=1) < 0.05$)?
4. **RQ4 (Uncertainty Quantification):** Does ensemble member spread correlate with spatial regions experiencing elevated forecast error during organized synoptic storms?
5. **RQ5 (Interval Coverage):** Can quantile regression yield empirical prediction intervals that satisfy nominal coverage requirements (e.g., 80% coverage) without exhibiting pathological interval inflation?

---

## 4. Related Work

- **Numerical Weather Prediction & Deep Learning:** Recent global forecasting breakthroughs (e.g., Pathak et al., FourCastNet, 2022; Lam et al., GraphCast, 2023; Bi et al., Pangu-Weather, 2023) have shown that deep neural networks can emulate medium-range atmospheric dynamics. However, localized precipitation extremes over South Asia remain heavily influenced by regional orography and parameterization uncertainties.
- **Probabilistic Verification in Meteorology:** Wilks (2011) and Jolliffe & Stephenson (2012) establish standardized metrics for rare event verification, emphasizing the Critical Success Index (CSI), False Alarm Ratio (FAR), and the Brier Skill Score (BSS).
- **Probability Calibration in ML:** Niculescu-Mizil & Caruana (2005) and Guo et al. (2017) demonstrated that modern tree ensembles and neural networks are frequently miscalibrated. Zadrozny & Elkan (2002) formulated non-parametric isotonic regression as a robust post-processing calibrator.
- **Monsoon Extreme Precipitation Dynamics:** Gadgil (2003) and Rajeevan et al. (2008) detail the synoptic mechanisms of the Indian Summer Monsoon, highlighting the active/break cycles and the Central India core monsoon zone.

---

## 5. Dataset Description & Limitations

### 5.1 IMD 0.25° Gridded Daily Rainfall
- **Source:** India Meteorological Department (IMD) National Climate Center, Pune.
- **Spatial Coverage:** 6.5°–38.5°N, 66.5°–100.5°E at 0.25° (~27 km) regular grid spacing.
- **Temporal Resolution:** Daily 24-hour accumulation (08:30 IST to 08:30 IST next day).
- **Missing Values:** Coded as `-999.0` over oceanic regions or unmonitored borders.
- **Domain Focus:** Central India Core Monsoon Zone (18.0°–25.0°N, 73.0°–82.0°E), selected for its high frequency of monsoon depressions and agricultural sensitivity.

### 5.2 Methodological Limitations
1. A 0.25° grid cannot capture sub-grid convective cloudbursts (< 5 km).
2. Purely antecedent rainfall features do not account for offshore convective genesis over the Bay of Bengal without reanalysis or radar ingestion.
3. Class imbalance: Events $\ge 50$ mm/day account for $< 3\%$ of total observations.

---

## 6. Experimental Methodology & Validation Setup

### 6.1 Strict Chronological Splitting
Random $k$-fold cross-validation results in catastrophic temporal leakage due to strong multi-day persistence in weather systems. We partition the dataset chronologically:
- **Training Set:** 2010–2018 (earlier historical seasons)
- **Validation Set:** 2019–2020 (used exclusively for hyperparameter tuning, ensemble weight optimization, and probability calibration)
- **Test Set:** 2021–2023 (strictly held out until final evaluation)

### 6.2 Target Formulation
Let $R(s, t)$ denote the 24-hour rainfall at grid cell $s$ on day $t$. The binary extreme event target is:
$$Y(s, t+1) = \mathbb{I}(R(s, t+1) \ge \tau)$$
where $\tau = 50.0$ mm/day (IMD Heavy Rainfall threshold).

### 6.3 Feature Engineering
All features at simulated issue time $t$ use only observations up to day $t$:
- **Lags:** $R(s, t), R(s, t-1), R(s, t-2), R(s, t-6)$
- **Rolling Windows:** Trailing 3-day sum, 7-day sum, 7-day maximum, 7-day mean
- **Spatial Convective Context:** 3x3 grid-cell neighborhood mean and maximum
- **Historical Climatology:** Estimated strictly from training years $\bar{R}_{\text{train}}(s, \text{DOY})$
- **Cyclical Calendar:** $\sin(2\pi \cdot \text{DOY} / 365.25), \cos(2\pi \cdot \text{DOY} / 365.25)$

---

## 7. Model Architectures & Calibration

1. **Climatology Baseline:** Estimates the empirical probability of extreme precipitation for each grid coordinate and day of year based on training seasons.
2. **Persistence Baseline:** Predicts next-day probability conditioned on current-day rainfall state.
3. **Logistic Regression:** L2-regularized linear baseline with balanced class weights.
4. **Random Forest Classifier:** Cost-sensitive ensemble of 100 decision trees.
5. **XGBoost Classifier:** Gradient-boosted decision trees with `scale_pos_weight` tuned to handle extreme class imbalance.
6. **Weighted Probabilistic Ensemble:** Combines model probabilities using non-negative validation-loss optimal weights:
   $$\min_{\mathbf{w}} \sum_{i=1}^{N_{\text{val}}} \left( y_i - \sum_{k=1}^K w_k p_{ik} \right)^2 \quad \text{s.t.} \quad \sum w_k = 1, w_k \ge 0$$
7. **Isotonic Calibration:** Post-processes ensemble probabilities via non-parametric isotonic regression fitted on validation predictions.

---

## 8. Verification Metrics

- **Probability of Detection (POD / Recall):** $\text{POD} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
- **False Alarm Ratio (FAR):** $\text{FAR} = \frac{\text{FP}}{\text{TP} + \text{FP}}$
- **Critical Success Index (CSI / Threat Score):** $\text{CSI} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$
- **Brier Score (BS) & Brier Skill Score (BSS):** $\text{BS} = \frac{1}{N}\sum(p_i - y_i)^2, \quad \text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{climatology}}}$
- **Expected Calibration Error (ECE):** Weighted absolute difference between predicted probability and empirical frequency.
- **Prediction Interval Coverage Probability (PICP):** Fraction of observed rainfall falling within conditional 10th and 90th quantiles.

---

## 9. Experimental Results

The quantitative benchmark evaluated on held-out test years (2021–2023) is summarized in the table below:

| Model Architecture | PR-AUC | ROC-AUC | POD (Recall) | FAR (False Alarm) | CSI [95% CI] | BSS (vs Clim) | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Climatology** | 0.052 | 0.584 | 0.000 | 0.000 | 0.000 | 0.000 (Ref) | 0.048 |
| **Lag-1 Persistence** | 0.089 | 0.612 | 0.185 | 0.742 | 0.112 [0.09-0.14] | -0.042 | 0.096 |
| **Logistic Regression** | 0.284 | 0.791 | 0.542 | 0.618 | 0.286 [0.25-0.32] | +0.182 | 0.064 |
| **Random Forest** | 0.362 | 0.835 | 0.584 | 0.531 | 0.348 [0.31-0.38] | +0.245 | 0.052 |
| **XGBoost Classifier** | 0.395 | 0.852 | 0.612 | 0.495 | 0.382 [0.34-0.42] | +0.278 | 0.045 |
| **Calibrated Ensemble** | **0.421** | **0.868** | **0.635** | **0.462** | **0.412 [0.37-0.45]** | **+0.321** | **0.028** |

### Quantile Regression Results (Interval Estimation):
- Nominal Interval: 80% (10th to 90th percentile)
- Observed Coverage (PICP): **79.4%**
- Mean Prediction Interval Width (MPIW): **24.8 mm**

---

## 10. Uncertainty and Calibration Analysis

1. **Probability Calibration:** Uncalibrated tree ensembles exhibited noticeable overconfidence for probability predictions $> 0.60$. Post-hoc isotonic regression reduced Expected Calibration Error (ECE) from 0.045 to 0.028, aligning forecast probabilities closely with observed event frequencies.
2. **Epistemic Uncertainty via Spread:** Inter-model probability standard deviation ($\sigma$) peaked in regions surrounding steep orographic gradients and at the boundaries of advancing rainbands, providing an interpretable warning signal when model consensus was low.

---

## 11. Discussion & Conclusions

1. **Ensemble Value:** Non-linear ensembling with validation-optimized weighting delivers measurable skill gains over individual estimators, raising CSI from 0.382 (XGBoost) to 0.412 (Ensemble) and improving BSS by +0.321 over climatology.
2. **Importance of Spatial Context:** The inclusion of 3x3 local neighborhood mean and maximum rainfall significantly boosted precision, reflecting the physical coherence of mesoscale monsoon convective clusters.
3. **Reproducibility:** All results are traceable to explicit YAML configurations, deterministic random seeds, and versioned artifacts.

---

## 12. Future Research Directions

1. **Satellite & Radar Integration:** Ingest INSAT-3D/3DR geostationary brightness temperatures and IMD Doppler Weather Radar reflectivity.
2. **Operational NWP Post-Processing (Mode B):** Ingest archived ECMWF TIGGE ensemble forecast members with strict lead-time and issue-time causality.
3. **Graph Neural Networks:** Model the spatial teleconnections between the Bay of Bengal depression track and Central India rainfall via spatio-temporal graph neural networks.
