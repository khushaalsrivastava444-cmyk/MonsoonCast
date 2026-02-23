"""Data preparation and feature engineering pipeline.

Usage:
    python scripts/prepare_data.py [--config configs/data.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monsooncast.config import load_data_config
from monsooncast.data.loaders import (
    ERA5Adapter,
    generate_synthetic_monsoon_dataset,
)
from monsooncast.data.preprocessing import align_and_preprocess_dataset
from monsooncast.evaluation.validation import chronological_split, verify_no_leakage
from monsooncast.features.engineering import (
    build_tabular_feature_dataset,
    compute_training_climatology,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Prepare features and chronological splits.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config YAML")
    parser.add_argument("--include-era5", action="store_true", help="Include coupled atmospheric reanalysis variables (ERA5)")
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    processed_dir = Path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Look for IMD files or fallback to synthetic fixture
    raw_files = list(Path(cfg.raw_dir).glob("imd/*.nc"))
    if raw_files:
        logger.info(f"Loading {len(raw_files)} raw IMD NetCDF files...")
        ds = xr.open_mfdataset(raw_files)
    else:
        sample_path = Path(cfg.raw_dir) / "synthetic_sample.nc"
        if not sample_path.exists():
            logger.info("No raw IMD NetCDF files detected. Generating synthetic test fixture...")
            ds = generate_synthetic_monsoon_dataset(
                output_path=sample_path,
                start_year=min(cfg.temporal_scope.train_years),
                end_year=max(cfg.temporal_scope.test_years),
                lat_bounds=(cfg.spatial_bounds.lat_min, cfg.spatial_bounds.lat_max),
                lon_bounds=(cfg.spatial_bounds.lon_min, cfg.spatial_bounds.lon_max),
                resolution=0.5
            )
        else:
            logger.info(f"Loading existing test fixture: {sample_path}")
            ds = xr.open_dataset(sample_path)

    # 2. Subset spatial bounds and filter monsoon season (JJAS)
    logger.info("Aligning spatial domain and filtering monsoon season (JJAS)...")
    ds_monsoon = align_and_preprocess_dataset(ds, cfg.spatial_bounds, cfg.temporal_scope)

    # 3. Compute Training Climatology strictly on train years (No Leakage!)
    train_years = cfg.temporal_scope.train_years
    times = pd.to_datetime(ds_monsoon["time"].values)
    train_mask = times.year.isin(train_years)
    ds_train = ds_monsoon.isel(time=train_mask)
    logger.info(f"Computing historical climatology strictly from training seasons ({min(train_years)}-{max(train_years)})...")
    train_climatology = compute_training_climatology(ds_train)

    # 4. Feature Engineering
    era5_ds = None
    if args.include_era5:
        logger.info("Coupling atmospheric reanalysis predictors (ERA5)...")
        era5_adapter = ERA5Adapter(data_dir=Path(cfg.raw_dir) / "era5")
        era5_ds = era5_adapter.load_variables(rainfall_ds=ds_monsoon)

    logger.info("Constructing lagged, rolling, and spatial neighborhood features...")
    df_features = build_tabular_feature_dataset(
        ds=ds_monsoon,
        train_climatology=train_climatology,
        era5_ds=era5_ds,
        threshold=cfg.target.default_threshold,
        lead_days=cfg.target.lead_days
    )

    # 5. Chronological Splitting
    logger.info("Executing strict chronological split...")
    df_train, df_val, df_test = chronological_split(
        df_features,
        train_years=cfg.temporal_scope.train_years,
        val_years=cfg.temporal_scope.val_years,
        test_years=cfg.temporal_scope.test_years
    )

    # 6. Leakage Verification
    feature_cols = [c for c in df_features.columns if c not in [
        "time", "lat", "lon", "year", "month", "doy", "target_rainfall", "target_event"
    ]]
    logger.info(f"Verifying leakage on {len(feature_cols)} predictor features...")
    verify_no_leakage(df_train, df_val, df_test, feature_cols=feature_cols, target_col="target_event")

    # 7. Save Processed Artifacts
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    logger.info(f"Processed datasets successfully saved to {processed_dir}:")
    logger.info(f"  - Train: {len(df_train)} rows -> {train_path}")
    logger.info(f"  - Val:   {len(df_val)} rows -> {val_path}")
    logger.info(f"  - Test:  {len(df_test)} rows -> {test_path}")


if __name__ == "__main__":
    main()
