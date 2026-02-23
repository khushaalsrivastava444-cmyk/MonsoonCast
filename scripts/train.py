"""Model training pipeline for baselines, machine learning, and calibrated ensembles.

Usage:
    python scripts/train.py [--config configs/baseline.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monsooncast.config import load_config
from monsooncast.models.baselines import (
    LogisticRegressionBaseline,
    PersistenceBaseline,
    SeasonalClimatologyBaseline,
)
from monsooncast.models.classical import (
    RandomForestExtremeClassifier,
    XGBoostExtremeClassifier,
    XGBoostQuantileRegressor,
)
from monsooncast.models.ensemble import WeightedProbabilityEnsemble
from monsooncast.models.neural import HAS_TORCH, PyTorchMLPClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train MonsoonCast models and ensemble.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to experiment config YAML")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/models", help="Directory to save model artifacts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    proc_dir = Path(args.processed_dir)
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Parquet partitions
    train_path = proc_dir / "train.parquet"
    val_path = proc_dir / "val.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Processed data not found in {proc_dir}. Please run 'python scripts/prepare_data.py' first."
        )

    logger.info(f"Loading training data from {train_path} and validation data from {val_path}...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    feature_cols = [c for c in df_train.columns if c not in [
        "time", "issue_date", "target_valid_date", "lat", "lon", "year", "month", "doy", "target_rainfall", "target_event"
    ]]
    y_train = df_train["target_event"].values
    y_train_amount = df_train["target_rainfall"].values
    y_val = df_val["target_event"].values

    logger.info(f"Feature set size: {len(feature_cols)} features. Extreme event frequency in train: {np.mean(y_train)*100:.2f}%")

    models_dict = {}

    # 2. Fit Baselines
    logger.info("Fitting Climatology baseline...")
    clim_model = SeasonalClimatologyBaseline().fit(df_train)
    models_dict["climatology"] = clim_model

    logger.info("Fitting Persistence baseline...")
    pers_model = PersistenceBaseline(threshold=cfg.data.target.default_threshold if cfg.data else 50.0).fit(df_train)
    models_dict["persistence"] = pers_model

    logger.info("Fitting Logistic Regression baseline...")
    log_model = LogisticRegressionBaseline().fit(df_train, y_train, feature_cols=feature_cols)
    models_dict["logistic_regression"] = log_model

    # 3. Fit Classical ML Models
    logger.info("Fitting Random Forest classifier...")
    rf_model = RandomForestExtremeClassifier(n_estimators=100, max_depth=8).fit(df_train, y_train, feature_cols=feature_cols)
    models_dict["random_forest"] = rf_model

    logger.info("Fitting XGBoost classifier...")
    xgb_model = XGBoostExtremeClassifier(n_estimators=100, max_depth=5, scale_pos_weight=5.0).fit(
        df_train, y_train, feature_cols=feature_cols, eval_set=[(df_val, y_val)]
    )
    models_dict["xgboost_classifier"] = xgb_model

    # 4. Optional PyTorch MLP
    if HAS_TORCH:
        try:
            logger.info("Fitting PyTorch Tabular MLP classifier with Monte Carlo Dropout...")
            torch_model = PyTorchMLPClassifier(hidden_dim=64, epochs=10).fit(df_train, y_train, feature_cols=feature_cols)
            models_dict["pytorch_mlp"] = torch_model
        except Exception as e:
            logger.warning(f"Could not fit PyTorch MLP: {e}. Continuing with tree models.")

    # 5. Fit Quantile Regressor for Rainfall Amounts
    logger.info("Fitting XGBoost Quantile Regressors (quantiles=[0.10, 0.50, 0.90])...")
    q_reg = XGBoostQuantileRegressor(quantiles=[0.10, 0.50, 0.90], n_estimators=60).fit(
        df_train, y_train_amount, feature_cols=feature_cols
    )
    models_dict["quantile_regressor"] = q_reg

    # 6. Build and Calibrate Ensemble strictly on Validation Partition
    logger.info("Assembling and calibrating probabilistic ensemble strictly on validation partition...")
    ensemble_members = {
        "logistic_regression": log_model,
        "random_forest": rf_model,
        "xgboost_classifier": xgb_model,
    }
    ensemble = WeightedProbabilityEnsemble(
        models=ensemble_members,
        weighting_scheme="val_optimized",
        calibration_method="isotonic"
    )
    ensemble.fit_weights_and_calibration(df_val, y_val)
    models_dict["ensemble"] = ensemble

    # 7. Serialize Models to Artifacts
    artifact_payload = {
        "models": models_dict,
        "feature_cols": feature_cols,
        "experiment_id": cfg.id,
        "config_hash": cfg.compute_hash(),
    }
    model_artifact_path = art_dir / "monsooncast_models.joblib"
    joblib.dump(artifact_payload, model_artifact_path)
    logger.info(f"Model artifacts successfully serialized to: {model_artifact_path}")


if __name__ == "__main__":
    main()
