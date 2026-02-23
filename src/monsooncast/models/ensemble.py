"""Weighted Probabilistic Ensemble and Uncertainty Quantification.

Combines individual model probabilities into a calibrated consensus forecast,
computing member spread and epistemic disagreement uncertainty.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from monsooncast.calibration.probability import ProbabilityCalibrator

logger = logging.getLogger(__name__)


class WeightedProbabilityEnsemble:
    """Ensemble combining probability forecasts from multiple base estimators.

    Features:
    - Equal weighting baseline.
    - Validation-optimized non-negative weights (constrained Brier score minimization).
    - Post-processing probability calibration (strictly fitted on validation split).
    - Predictive uncertainty quantification: ensemble spread and entropy.
    """

    def __init__(
        self,
        models: dict[str, Any],
        weighting_scheme: str = "val_optimized",
        calibration_method: str = "isotonic",
    ):
        self.models = models
        self.member_names = list(models.keys())
        self.weighting_scheme = weighting_scheme
        self.weights = np.ones(len(self.member_names)) / len(self.member_names)
        self.calibrator = ProbabilityCalibrator(method=calibration_method)
        self.is_fitted = False

    def fit_weights_and_calibration(
        self, X_val: pd.DataFrame, y_val: np.ndarray
    ) -> WeightedProbabilityEnsemble:
        """Fit ensemble weights and probability calibration strictly on validation data."""
        y_val_arr = np.asarray(y_val, dtype=float)

        # 1. Collect predictions from all members on validation set
        member_preds = []
        for name in self.member_names:
            p_val = self.models[name].predict_proba(X_val)
            member_preds.append(p_val)
        P_val = np.column_stack(member_preds)  # shape: (N_val, K)

        # 2. Determine weights
        if self.weighting_scheme == "val_optimized":
            K = len(self.member_names)

            # Objective: Minimize Brier Score on validation set
            def brier_loss(w):
                p_ens = np.dot(P_val, w)
                return np.mean((p_ens - y_val_arr) ** 2)

            init_w = np.ones(K) / K
            bounds = [(0.0, 1.0) for _ in range(K)]
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            res = minimize(
                brier_loss, init_w, method="SLSQP", bounds=bounds, constraints=constraints
            )
            if res.success:
                self.weights = res.x
                logger.info(
                    f"Optimized ensemble weights: {dict(zip(self.member_names, np.round(self.weights, 3)))}"
                )
            else:
                logger.warning(
                    "Weight optimization did not converge; falling back to equal weights."
                )
                self.weights = init_w
        else:
            self.weights = np.ones(len(self.member_names)) / len(self.member_names)

        # 3. Fit probability calibrator on validation ensemble predictions
        raw_ens_val = np.dot(P_val, self.weights)
        self.calibrator.fit(raw_ens_val, y_val_arr)
        self.is_fitted = True
        return self

    def predict_components(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return individual probability predictions from each ensemble member."""
        components = {}
        for name in self.member_names:
            components[name] = self.models[name].predict_proba(X)
        return components

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated consensus probability."""
        components = self.predict_components(X)
        P = np.column_stack([components[name] for name in self.member_names])
        raw_prob = np.dot(P, self.weights)
        calibrated_prob = self.calibrator.transform(raw_prob)
        return np.clip(calibrated_prob, 0.0, 1.0)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict calibrated probability along with member spread and disagreement entropy.

        Returns:
        - 'calibrated_prob': Calibrated ensemble consensus probability
        - 'raw_prob': Uncalibrated weighted probability
        - 'spread': Inter-member standard deviation (epistemic spread)
        - 'entropy': Predictive binary Shannon entropy
        - 'member_probs': Dict of individual member probabilities
        """
        components = self.predict_components(X)
        P = np.column_stack([components[name] for name in self.member_names])

        raw_prob = np.dot(P, self.weights)
        calibrated_prob = self.calibrator.transform(raw_prob)

        # Member spread (weighted or unweighted standard deviation)
        spread = np.std(P, axis=1)

        # Binary entropy: H(p) = -p log2(p) - (1-p) log2(1-p)
        p_safe = np.clip(calibrated_prob, 1e-7, 1.0 - 1e-7)
        entropy = -p_safe * np.log2(p_safe) - (1.0 - p_safe) * np.log2(1.0 - p_safe)

        return {
            "calibrated_prob": calibrated_prob,
            "raw_prob": raw_prob,
            "spread": spread,
            "entropy": entropy,
            "member_probs": components,
        }
