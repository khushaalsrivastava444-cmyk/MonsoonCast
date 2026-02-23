# MonsoonCast Research Methodology

## 1. Problem Formulation

Let $R(s, t)$ denote the 24-hour accumulated rainfall (mm/day) at spatial grid coordinate $s = (\text{lat}, \text{lon})$ on day $t$.

We formulate two complementary prediction problems:

### Problem 1: Probabilistic Extreme-Event Classification
Given meteorological and antecedent rainfall observations up to day $t$, estimate the conditional probability of an extreme rainfall event occurring on the subsequent day $t+1$:

$$P(Y(s, t+1) = 1 \mid \mathcal{F}_t)$$

where the binary extreme indicator is:

$$Y(s, t+1) = \mathbb{I}(R(s, t+1) \ge \tau)$$

with $\tau \in \{35.5, 50.0, 64.5, 100.0\}$ mm/day, or a local grid-cell percentile threshold $\tau = q_{95}(s)$ computed solely over the training years.

### Problem 2: Conditional Rainfall Quantile Estimation
Predict conditional quantiles $Q_\alpha(R(s, t+1) \mid \mathcal{F}_t)$ for risk levels $\alpha \in \{0.10, 0.50, 0.90\}$ to provide calibrated prediction intervals:

$$I_{0.80}(s, t+1) = [Q_{0.10}(s, t+1), Q_{0.90}(s, t+1)]$$

## 2. Experimental Modes

### Mode A: Antecedent Rainfall and Diagnostic Reanalysis
Predicts $Y(s, t+1)$ using strictly lagged observations available at the simulated issue time (08:30 IST on day $t$):
- Rainfall lags: $R(s, t), R(s, t-1), R(s, t-2), R(s, t-6)$
- Trailing cumulative sums: $\sum_{k=0}^2 R(s, t-k)$ (3-day total), $\sum_{k=0}^6 R(s, t-k)$ (7-day total)
- Local spatial neighborhood aggregates: $3 \times 3$ grid-cell mean and maximum
- Climatological normal: $\bar{R}_{\text{train}}(s, \text{DOY})$
- Atmospheric reanalysis predictors (when Mode A is run retrospectively, availability times are documented explicitly)

### Mode B: Forecast-Informed Ensemble Post-Processing
When numerical weather prediction (NWP) ensemble member forecasts are ingested, the pipeline strictly tracks:
- Forecast Initialization Time $t_{\text{init}}$
- Lead Time $\Delta t$ (e.g. 24h, 48h)
- Valid Time $t_{\text{valid}} = t_{\text{init}} + \Delta t$
- Ensemble Member ID $m \in \{1, \dots, M\}$

## 3. Strict Validation & Leakage Prevention

1. **Chronological Splitting:**
   - Training Partition: Earlier historical monsoon seasons (June–September)
   - Validation Partition: Subsequent middle monsoon seasons
   - Test Partition: Held-out recent seasons
2. **Feature Imputation & Standardization:**
   - Imputer medians and standard scalers are computed strictly on the training partition.
3. **Climatology Isolation:**
   - Climatology baseline frequencies $\pi(s, \text{DOY})$ are estimated strictly on training seasons.
4. **Calibration Isolation:**
   - Platt scaling (logistic) and isotonic calibrators are fitted strictly on the validation partition.

## 4. Meteorological Evaluation Metrics

- **Probability of Detection (POD) / Recall:**
  $$\text{POD} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **False Alarm Ratio (FAR):**
  $$\text{FAR} = \frac{\text{FP}}{\text{TP} + \text{FP}}$$
- **Critical Success Index (CSI) / Threat Score:**
  $$\text{CSI} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
- **Brier Score (BS) & Brier Skill Score (BSS):**
  $$\text{BS} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2, \quad \text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{climatology}}}$$
- **Reliability & Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **Block-Resampled Uncertainty:**
  To account for strong spatial and temporal autocorrelation across monsoon systems, metric 95% confidence intervals are estimated using temporal block bootstrapping.
