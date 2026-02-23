"""MonsoonCast Model Registry: Baselines, Classical ML, Neural, and Ensembles."""

from monsooncast.models.baselines import (
    LogisticRegressionBaseline,
    PersistenceBaseline,
    RidgeRainfallRegressor,
    SeasonalClimatologyBaseline,
)
from monsooncast.models.classical import (
    RandomForestExtremeClassifier,
    XGBoostExtremeClassifier,
    XGBoostQuantileRegressor,
)
from monsooncast.models.ensemble import (
    WeightedProbabilityEnsemble,
)
from monsooncast.models.neural import (
    HAS_TORCH,
    PyTorchMLPClassifier,
)

__all__ = [
    "HAS_TORCH",
    "LogisticRegressionBaseline",
    "PersistenceBaseline",
    "PyTorchMLPClassifier",
    "RandomForestExtremeClassifier",
    "RidgeRainfallRegressor",
    "SeasonalClimatologyBaseline",
    "WeightedProbabilityEnsemble",
    "XGBoostExtremeClassifier",
    "XGBoostQuantileRegressor",
]
