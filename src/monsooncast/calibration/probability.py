"""Post-processing probability calibration using Platt scaling and Isotonic regression.

Strict Validation Protocol:
- Calibration models must be fit EXCLUSIVELY on validation predictions.
- Never fit or select calibrators using held-out test data.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class PlattScaler:
    """Parametric sigmoid probability calibration (Platt 1999).

    Fits a 1D logistic regression model: P(Y=1 | p) = 1 / (1 + exp(-(a * logit(p) + b)))
    """

    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> PlattScaler:
        p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        y = np.asarray(y_true, dtype=int)

        # Logit transform
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        self.model.fit(logits, y)
        self.is_fitted = True
        logger.info(
            f"Fitted Platt Scaler: a={self.model.coef_[0][0]:.4f}, b={self.model.intercept_[0]:.4f}"
        )
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PlattScaler must be fitted before predict_proba.")
        p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        calibrated_p = self.model.predict_proba(logits)[:, 1]
        return np.clip(calibrated_p, 0.0, 1.0)


class IsotonicCalibrator:
    """Non-parametric monotonic probability calibration (Zadrozny & Elkan 2002).

    Fits a step-wise monotonic non-decreasing mapping between raw probabilities and empirical outcomes.
    """

    def __init__(self):
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        y = np.asarray(y_true, dtype=float)
        self.model.fit(p, y)
        self.is_fitted = True
        logger.info("Fitted Isotonic Calibrator.")
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator must be fitted before predict_proba.")
        p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        calibrated_p = self.model.predict(p)
        return np.clip(calibrated_p, 0.0, 1.0)


class ProbabilityCalibrator:
    """Unified calibrator interface supporting 'platt', 'isotonic', and 'none'."""

    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        if self.method == "platt":
            self.calibrator = PlattScaler()
        elif self.method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        elif self.method in ("none", "identity", "raw"):
            self.calibrator = None
        else:
            raise ValueError(
                f"Unknown calibration method: '{method}'. Choose 'platt', 'isotonic', or 'none'."
            )

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator:
        if self.calibrator is not None:
            self.calibrator.fit(y_prob, y_true)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        return self.calibrator.predict_proba(y_prob)
