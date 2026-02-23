"""MonsoonCast: Research-Grade AI Ensemble Framework for Extreme Monsoon Rainfall Prediction

and Uncertainty Quantification over India.
"""

__version__ = "0.1.0"
__author__ = "Khushaal Srivastava"
__date__ = "2026-02-23"

from monsooncast.config import (
    DataConfig,
    ExperimentConfig,
    load_config,
)

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "__version__",
    "load_config",
]
