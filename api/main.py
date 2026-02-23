"""FastAPI Application for MonsoonCast Model Serving and Experiment Exploration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monsooncast_api")

app = FastAPI(
    title="MonsoonCast Research API",
    description="Scientific API for extreme monsoon rainfall prediction, probabilistic calibration, and uncertainty quantification over India.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model cache
MODEL_CACHE = {}
MODEL_PATH = Path("artifacts/models/monsooncast_models.joblib")


def get_models():
    """Load and cache serialized model artifacts."""
    if "models" not in MODEL_CACHE:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MonsoonCast model artifacts not found. Please train models first using 'python scripts/train.py'."
            )
        logger.info(f"Loading models from {MODEL_PATH} into API cache...")
        payload = joblib.load(MODEL_PATH)
        MODEL_CACHE["payload"] = payload
        MODEL_CACHE["models"] = payload["models"]
        MODEL_CACHE["feature_cols"] = payload["feature_cols"]
    return MODEL_CACHE["models"], MODEL_CACHE["feature_cols"]


# Request / Response Schemas
class GridPointFeatures(BaseModel):
    lat: float = Field(..., ge=6.0, le=39.0, description="Latitude coordinate (degrees North)")
    lon: float = Field(..., ge=66.0, le=101.0, description="Longitude coordinate (degrees East)")
    doy: int = Field(..., ge=1, le=366, description="Day of year (1-366)")
    rain_current: float = Field(..., ge=0.0, description="Rainfall on forecast issue day t (mm/day)")
    rain_lag1: float = Field(default=0.0, ge=0.0, description="Rainfall at day t-1 (mm/day)")
    rain_lag2: float = Field(default=0.0, ge=0.0, description="Rainfall at day t-2 (mm/day)")
    rain_lag3: float = Field(default=0.0, ge=0.0, description="Rainfall at day t-3 (mm/day)")
    rain_lag7: float = Field(default=0.0, ge=0.0, description="Rainfall at day t-6 (mm/day)")
    rain_roll_sum3: float = Field(default=0.0, ge=0.0, description="Trailing 3-day sum (mm)")
    rain_roll_sum7: float = Field(default=0.0, ge=0.0, description="Trailing 7-day sum (mm)")
    rain_roll_max7: float = Field(default=0.0, ge=0.0, description="Trailing 7-day max (mm/day)")
    rain_roll_mean7: float = Field(default=0.0, ge=0.0, description="Trailing 7-day mean (mm/day)")
    rain_spatial_mean3x3: float = Field(default=0.0, ge=0.0, description="3x3 local neighborhood mean (mm/day)")
    rain_spatial_max3x3: float = Field(default=0.0, ge=0.0, description="3x3 local neighborhood max (mm/day)")
    climatology_mean: float = Field(default=0.0, ge=0.0, description="Historical climatological mean (mm/day)")
    doy_sin: Optional[float] = None
    doy_cos: Optional[float] = None


class EventProbabilityRequest(BaseModel):
    samples: List[GridPointFeatures]
    model_name: str = Field(default="ensemble", description="Model name: 'ensemble', 'xgboost_classifier', 'random_forest', 'logistic_regression'")


class EventProbabilityResponse(BaseModel):
    model_name: str
    calibrated_probability: List[float]
    epistemic_spread: Optional[List[float]] = None
    disagreement_entropy: Optional[List[float]] = None
    member_predictions: Optional[Dict[str, List[float]]] = None
    threshold_mm: float = 50.0
    status: str = "success"


class QuantileRequest(BaseModel):
    samples: List[GridPointFeatures]


class QuantileResponse(BaseModel):
    quantiles: Dict[str, List[float]]
    status: str = "success"


@app.get("/health", tags=["System"])
def health_check():
    """Health status and artifact check."""
    return {
        "status": "healthy",
        "service": "MonsoonCast Research API",
        "version": "0.1.0",
        "artifacts_available": MODEL_PATH.exists(),
        "release_date": "2026-02-23"
    }


@app.get("/metadata", tags=["System"])
def get_metadata():
    """Retrieve metadata about supported spatial domain, models, and thresholds."""
    return {
        "domain": {
            "name": "Central India Core Monsoon Zone",
            "lat_bounds": [18.0, 25.0],
            "lon_bounds": [73.0, 82.0],
            "resolution_deg": 0.25,
            "season": "Summer Monsoon (JJAS: June-September)"
        },
        "target": {
            "definition": "Next-day 24h rainfall >= 50 mm/day (IMD Heavy Rainfall)",
            "lead_time": "1-day (24 hours)"
        },
        "supported_models": [
            "climatology",
            "persistence",
            "logistic_regression",
            "random_forest",
            "xgboost_classifier",
            "ensemble"
        ],
        "calibration": "Isotonic Regression & Platt Scaling"
    }


@app.get("/experiments", tags=["Experiments"])
def list_experiments():
    """List all registered experiment runs."""
    exp_dir = Path("artifacts/experiments")
    if not exp_dir.exists():
        return {"experiments": []}
    files = list(exp_dir.glob("*.json"))
    exps = []
    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
            exps.append({
                "experiment_id": data.get("experiment_id", f.stem),
                "timestamp": data.get("timestamp"),
                "config_hash": data.get("config_hash")
            })
    return {"experiments": exps}


@app.get("/experiments/{experiment_id}", tags=["Experiments"])
def get_experiment(experiment_id: str):
    """Retrieve details and configuration of a specific experiment run."""
    path = Path("artifacts/experiments") / f"{experiment_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    with open(path, "r") as fp:
        return json.load(fp)


@app.get("/metrics/{experiment_id}", tags=["Experiments"])
def get_experiment_metrics(experiment_id: str):
    """Retrieve held-out test evaluation metrics for a specific experiment."""
    path = Path("artifacts/experiments") / f"{experiment_id}.json"
    if not path.exists():
        # Fallback to general metrics if exact experiment ID isn't isolated
        gen_path = Path("artifacts/metrics/evaluation_results.json")
        if gen_path.exists():
            with open(gen_path, "r") as fp:
                return json.load(fp)
        raise HTTPException(status_code=404, detail=f"Metrics for experiment '{experiment_id}' not found.")
    with open(path, "r") as fp:
        data = json.load(fp)
        return data.get("metrics_summary", {})


@app.post("/predict/event-probability", response_model=EventProbabilityResponse, tags=["Inference"])
def predict_event_probability(req: EventProbabilityRequest):
    """Predict extreme rainfall event probability with predictive uncertainty quantification."""
    models, feature_cols = get_models()
    
    # Build dataframe from incoming samples
    rows = []
    for s in req.samples:
        d = s.model_dump()
        if d.get("doy_sin") is None:
            d["doy_sin"] = np.sin(2.0 * np.pi * d["doy"] / 365.25)
        if d.get("doy_cos") is None:
            d["doy_cos"] = np.cos(2.0 * np.pi * d["doy"] / 365.25)
        rows.append(d)
    df_in = pd.DataFrame(rows)
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df_in.columns:
            df_in[col] = 0.0

    if req.model_name == "ensemble":
        ens = models["ensemble"]
        unc_res = ens.predict_with_uncertainty(df_in)
        
        member_preds_dict = {
            k: [round(float(v), 4) for v in vals]
            for k, vals in unc_res["member_probs"].items()
        }
        
        return EventProbabilityResponse(
            model_name="ensemble",
            calibrated_probability=[round(float(p), 4) for p in unc_res["calibrated_prob"]],
            epistemic_spread=[round(float(s), 4) for s in unc_res["spread"]],
            disagreement_entropy=[round(float(e), 4) for e in unc_res["entropy"]],
            member_predictions=member_preds_dict,
            threshold_mm=50.0
        )
    elif req.model_name in models:
        m = models[req.model_name]
        probs = m.predict_proba(df_in)
        return EventProbabilityResponse(
            model_name=req.model_name,
            calibrated_probability=[round(float(p), 4) for p in probs],
            threshold_mm=50.0
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model_name}' not available. Choose from: {list(models.keys())}"
        )


@app.post("/predict/rainfall-quantiles", response_model=QuantileResponse, tags=["Inference"])
def predict_rainfall_quantiles(req: QuantileRequest):
    """Predict conditional rainfall amount quantiles (10%, 50%, 90%) for interval estimation."""
    models, feature_cols = get_models()
    if "quantile_regressor" not in models:
        raise HTTPException(status_code=404, detail="Quantile regressor model not trained.")
        
    rows = []
    for s in req.samples:
        d = s.model_dump()
        if d.get("doy_sin") is None:
            d["doy_sin"] = np.sin(2.0 * np.pi * d["doy"] / 365.25)
        if d.get("doy_cos") is None:
            d["doy_cos"] = np.cos(2.0 * np.pi * d["doy"] / 365.25)
        rows.append(d)
    df_in = pd.DataFrame(rows)
    for col in feature_cols:
        if col not in df_in.columns:
            df_in[col] = 0.0

    q_reg = models["quantile_regressor"]
    q_preds = q_reg.predict_quantiles(df_in)
    
    out_dict = {}
    for q_val, arr in q_preds.items():
        out_dict[f"q_{int(q_val*100):02d}"] = [round(float(x), 2) for x in arr]
        
    return QuantileResponse(quantiles=out_dict)
