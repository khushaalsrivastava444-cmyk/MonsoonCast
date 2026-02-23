"""Configuration management using Pydantic schemas and YAML parsing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SpatialBounds(BaseModel):
    lat_min: float = Field(default=18.0, ge=5.0, le=40.0)
    lat_max: float = Field(default=25.0, ge=5.0, le=40.0)
    lon_min: float = Field(default=73.0, ge=65.0, le=105.0)
    lon_max: float = Field(default=82.0, ge=65.0, le=105.0)


class TemporalScope(BaseModel):
    monsoon_months: list[int] = Field(default=[6, 7, 8, 9])
    train_years: list[int] = Field(default_factory=lambda: list(range(2010, 2019)))
    val_years: list[int] = Field(default_factory=lambda: [2019, 2020])
    test_years: list[int] = Field(default_factory=lambda: [2021, 2022, 2023])


class TargetConfig(BaseModel):
    lead_days: int = Field(default=1, ge=1)
    default_threshold: float = Field(default=50.0, gt=0.0)
    threshold_options: list[float] = Field(default=[35.5, 50.0, 64.5, 100.0])
    percentile_threshold: float = Field(default=95.0, ge=50.0, lt=100.0)


class IMDConfig(BaseModel):
    file_pattern: str = "data/raw/imd/rainfall_*.nc"
    var_name: str = "rain"
    fill_value: float = -999.0
    units: str = "mm/day"


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    imd: IMDConfig = Field(default_factory=IMDConfig)
    spatial_bounds: SpatialBounds = Field(default_factory=SpatialBounds)
    temporal_scope: TemporalScope = Field(default_factory=TemporalScope)
    target: TargetConfig = Field(default_factory=TargetConfig)


class FeatureConfig(BaseModel):
    lags: list[int] = Field(default=[1, 2, 3, 7])
    rolling_sums: list[int] = Field(default=[3, 7])
    rolling_max: list[int] = Field(default=[7])
    rolling_mean: list[int] = Field(default=[7])
    include_spatial_neighbors: bool = True
    include_climatology: bool = True
    include_cyclical_calendar: bool = True


class ModelHyperparameters(BaseModel):
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    enabled: bool = True
    method: str = "isotonic"  # 'platt' or 'isotonic'
    fit_partition: str = "val"


class EnsembleConfig(BaseModel):
    members: list[str] = Field(
        default=["logistic_regression", "random_forest", "xgboost_classifier"]
    )
    weighting_scheme: str = "val_optimized"  # 'equal', 'val_optimized'
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    quantile_regression: dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True, "quantiles": [0.1, 0.5, 0.9]}
    )


class EvaluationConfig(BaseModel):
    metric_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    optimize_threshold_on_val: bool = True
    n_bootstrap_samples: int = Field(default=500, ge=10)


class ExperimentConfig(BaseModel):
    id: str = "monsooncast_default"
    description: str = "Default MonsoonCast Experiment"
    seed: int = 42
    data_config_path: str = "configs/data.yaml"
    data: DataConfig | None = None
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: dict[str, Any] = Field(default_factory=dict)
    ensemble: EnsembleConfig | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the experiment parameters."""
        serialized = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Safely load YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    """Load and validate DataConfig from YAML."""
    raw = load_yaml(path)
    data_data = raw.get("data", raw)
    return DataConfig.model_validate(data_data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load full ExperimentConfig with attached DataConfig."""
    raw = load_yaml(path)
    exp = ExperimentConfig.model_validate(raw.get("experiment", raw))

    # Load attached data config if path is provided
    data_cfg_path = raw.get("data_config", exp.data_config_path)
    if Path(data_cfg_path).exists():
        exp.data = load_data_config(data_cfg_path)
    elif exp.data is None:
        exp.data = DataConfig()

    return exp
