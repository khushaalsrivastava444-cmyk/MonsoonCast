"""Scientific baseline models for extreme rainfall prediction.

1. Seasonal Climatology Baseline: Empirical frequency of events by (lat, lon, DOY) on train set.
2. Persistence Baseline: Next-day probability conditioned on current-day rainfall state.
3. Logistic Regression: Linear benchmark with balanced class weighting.
4. Ridge Regression: Linear benchmark for continuous rainfall amounts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SeasonalClimatologyBaseline:
    """Predicts historical empirical extreme-event frequency conditioned on (lat, lon, DOY).

    Fit strictly on training partition to represent the climatological prior.
    """

    def __init__(self, smoothing_days: int = 7):
        self.smoothing_days = smoothing_days
        self.climatology_map: dict[tuple, float] = {}
        self.global_prior: float = 0.0
        self.is_fitted = False

    def fit(
        self, df_train: pd.DataFrame, target_col: str = "target_event"
    ) -> SeasonalClimatologyBaseline:
        y = df_train[target_col].values
        self.global_prior = float(np.mean(y)) if len(y) > 0 else 0.05

        # Group by (lat, lon, doy)
        grouped = df_train.groupby(["lat", "lon", "doy"])[target_col].mean().to_dict()
        self.climatology_map = grouped
        self.is_fitted = True
        logger.info(f"Fitted Climatology baseline (global event prior = {self.global_prior:.4f})")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        probs = np.empty(len(df), dtype=float)
        lats = df["lat"].values
        lons = df["lon"].values
        doys = df["doy"].values

        for i in range(len(df)):
            key = (lats[i], lons[i], doys[i])
            # Return cell-specific DOY climatology or fallback to global prior
            probs[i] = self.climatology_map.get(key, self.global_prior)

        return np.clip(probs, 0.0, 1.0)


class PersistenceBaseline:
    """Predicts next-day event probability conditioned on whether current day had rainfall."""

    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold
        self.prob_given_extreme: float = 0.0
        self.prob_given_non_extreme: float = 0.0
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame, target_col: str = "target_event") -> PersistenceBaseline:
        # Current rainfall
        rain_current = df_train["rain_current"].values
        is_extreme_today = rain_current >= self.threshold
        y = df_train[target_col].values

        if np.sum(is_extreme_today) > 0:
            self.prob_given_extreme = float(np.mean(y[is_extreme_today]))
        else:
            self.prob_given_extreme = 0.20

        if np.sum(~is_extreme_today) > 0:
            self.prob_given_non_extreme = float(np.mean(y[~is_extreme_today]))
        else:
            self.prob_given_non_extreme = 0.02

        self.is_fitted = True
        logger.info(
            f"Fitted Persistence baseline: P(Y=1|R_t>=50)={self.prob_given_extreme:.4f}, "
            f"P(Y=1|R_t<50)={self.prob_given_non_extreme:.4f}"
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        rain_current = df["rain_current"].values
        is_extreme = rain_current >= self.threshold
        probs = np.where(is_extreme, self.prob_given_extreme, self.prob_given_non_extreme)
        return np.clip(probs, 0.0, 1.0)


class LogisticRegressionBaseline:
    """L2-penalized Logistic Regression with feature standardization and class weighting."""

    def __init__(self, C: float = 1.0, class_weight: str = "balanced", max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C, class_weight=class_weight, max_iter=max_iter, solver="lbfgs"
        )
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray, feature_cols: list[str] | None = None
    ) -> LogisticRegressionBaseline:
        self.feature_cols = feature_cols or list(X_train.columns)
        X_mat = X_train[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X_mat)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        logger.info("Fitted Logistic Regression baseline.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        X_mat = X[self.feature_cols].values
        X_scaled = self.scaler.transform(X_mat)
        return self.model.predict_proba(X_scaled)[:, 1]


class RidgeRainfallRegressor:
    """Linear regression baseline for continuous rainfall amount estimation."""

    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray, feature_cols: list[str] | None = None
    ) -> RidgeRainfallRegressor:
        self.feature_cols = feature_cols or list(X_train.columns)
        X_mat = X_train[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X_mat)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        X_mat = X[self.feature_cols].values
        X_scaled = self.scaler.transform(X_mat)
        preds = self.model.predict(X_scaled)
        return np.clip(preds, 0.0, None)  # Physical constraint: rain >= 0
