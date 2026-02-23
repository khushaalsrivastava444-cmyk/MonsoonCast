"""Classical Machine Learning models: Random Forest and XGBoost classifiers & regressors."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


class RandomForestExtremeClassifier:
    """Random Forest Classifier with balanced class weights for rare extreme event prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 8,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray, feature_cols: list[str] | None = None
    ) -> RandomForestExtremeClassifier:
        self.feature_cols = feature_cols or list(X_train.columns)
        X_mat = X_train[self.feature_cols].values
        self.model.fit(X_mat, y_train)
        self.is_fitted = True
        logger.info(f"Fitted Random Forest Classifier with {self.model.n_estimators} trees.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        X_mat = X[self.feature_cols].values
        return self.model.predict_proba(X_mat)[:, 1]


class XGBoostExtremeClassifier:
    """XGBoost Classifier with scale_pos_weight adjustment for extreme event imbalance."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 5.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="logloss",
        )
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_cols: list[str] | None = None,
        eval_set: list | None = None,
    ) -> XGBoostExtremeClassifier:
        self.feature_cols = feature_cols or list(X_train.columns)
        X_mat = X_train[self.feature_cols].values

        xgb_eval_set = None
        if eval_set:
            xgb_eval_set = [(X_e[self.feature_cols].values, y_e) for X_e, y_e in eval_set]

        self.model.fit(X_mat, y_train, eval_set=xgb_eval_set, verbose=False)
        self.is_fitted = True
        logger.info("Fitted XGBoost Classifier.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        X_mat = X[self.feature_cols].values
        return self.model.predict_proba(X_mat)[:, 1]


class XGBoostQuantileRegressor:
    """XGBoost Gradient Boosted Quantile Regressor for conditional rainfall intervals."""

    def __init__(
        self,
        quantiles: list[float] = [0.10, 0.50, 0.90],
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.quantiles = sorted(quantiles)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models: dict[float, XGBRegressor] = {}
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray, feature_cols: list[str] | None = None
    ) -> XGBoostQuantileRegressor:
        self.feature_cols = feature_cols or list(X_train.columns)
        X_mat = X_train[self.feature_cols].values

        for q in self.quantiles:
            logger.info(f"Fitting XGBoost Quantile Regressor for quantile alpha={q:.2f}...")
            reg = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            reg.fit(X_mat, y_train)
            self.models[q] = reg

        self.is_fitted = True
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Models must be fitted before predict_quantiles.")
        X_mat = X[self.feature_cols].values
        preds = {}
        for q, model in self.models.items():
            # Physical constraint: rainfall quantiles cannot be negative
            preds[q] = np.clip(model.predict(X_mat), 0.0, None)
        return preds
