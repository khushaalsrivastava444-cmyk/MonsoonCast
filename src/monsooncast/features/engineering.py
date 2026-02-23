"""Feature engineering pipeline for extreme monsoon rainfall prediction.

Strict Causality Rules:
- All features generated for prediction issue day t use ONLY data available up to day t.
- Rolling windows explicitly look backward (e.g. t, t-1, ... t-w+1).
- Target is the rainfall or event on day t + lead_days (default t+1).
- Training climatology is estimated strictly on the training partition years.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter, uniform_filter

from monsooncast.config import FeatureConfig

logger = logging.getLogger(__name__)


def compute_training_climatology(train_ds: xr.Dataset, var_name: str = "rain") -> xr.DataArray:
    """Compute empirical daily climatology by (DOY, lat, lon) exclusively on training years."""
    rain = train_ds[var_name]
    # Group by day of year
    climatology = rain.groupby("time.dayofyear").mean(dim="time", skipna=True)
    climatology.attrs["description"] = "Daily mean rainfall computed strictly on training years"
    return climatology


def compute_spatial_neighborhood_features(
    ds: xr.Dataset, var_name: str = "rain"
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute 3x3 local neighborhood mean and max on the 2D spatial grid for each day.

    Captures organized mesoscale convective systems passing across adjacent grid cells.
    """
    rain = ds[var_name].values  # shape: (time, lat, lon)
    n_time, n_lat, n_lon = rain.shape

    mean_feat = np.empty_like(rain)
    max_feat = np.empty_like(rain)

    for t in range(n_time):
        frame = rain[t]
        # Treat NaNs as 0 for neighborhood filter, then mask back
        nan_mask = np.isnan(frame)
        frame_clean = np.where(nan_mask, 0.0, frame)

        # 3x3 spatial filter
        spatial_mean = uniform_filter(frame_clean, size=3, mode="reflect")
        spatial_max = maximum_filter(frame_clean, size=3, mode="reflect")

        mean_feat[t] = np.where(nan_mask, np.nan, spatial_mean)
        max_feat[t] = np.where(nan_mask, np.nan, spatial_max)

    da_mean = xr.DataArray(mean_feat, coords=ds[var_name].coords, dims=ds[var_name].dims)
    da_max = xr.DataArray(max_feat, coords=ds[var_name].coords, dims=ds[var_name].dims)
    return da_mean, da_max


def compute_lagged_features(
    series: pd.Series, lags: list[int] = [1, 2, 3, 7], group_col: pd.Series | None = None
) -> pd.DataFrame:
    """Compute backward-shifted lags for a 1D time series at a single grid cell.

    If group_col is provided (e.g. series.index.year), shifts are strictly confined within
    each group to prevent temporal boundary leakage across non-contiguous seasons.
    """
    lag_dict = {}
    if group_col is not None:
        grouped = series.groupby(group_col)
        for lag in lags:
            lag_dict[f"rain_lag{lag}"] = grouped.shift(lag)
    else:
        for lag in lags:
            lag_dict[f"rain_lag{lag}"] = series.shift(lag)
    return pd.DataFrame(lag_dict, index=series.index)


def compute_rolling_features(
    series: pd.Series,
    rolling_sums: list[int] = [3, 7],
    rolling_max: list[int] = [7],
    rolling_mean: list[int] = [7],
    group_col: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute backward-looking rolling statistics on a 1D rainfall series.

    Uses min_periods=1 to allow graceful handling of early-season boundary.
    All rolling windows include the current day observation R(t) and past days.
    If group_col is provided, rolling windows do not bleed across season boundaries.
    """
    roll_dict = {}
    if group_col is not None:
        grouped = series.groupby(group_col)
        for w in rolling_sums:
            roll_dict[f"rain_roll_sum{w}"] = grouped.transform(
                lambda s: s.rolling(window=w, min_periods=1).sum()
            )
        for w in rolling_max:
            roll_dict[f"rain_roll_max{w}"] = grouped.transform(
                lambda s: s.rolling(window=w, min_periods=1).max()
            )
        for w in rolling_mean:
            roll_dict[f"rain_roll_mean{w}"] = grouped.transform(
                lambda s: s.rolling(window=w, min_periods=1).mean()
            )
    else:
        for w in rolling_sums:
            roll_dict[f"rain_roll_sum{w}"] = series.rolling(window=w, min_periods=1).sum()
        for w in rolling_max:
            roll_dict[f"rain_roll_max{w}"] = series.rolling(window=w, min_periods=1).max()
        for w in rolling_mean:
            roll_dict[f"rain_roll_mean{w}"] = series.rolling(window=w, min_periods=1).mean()
    return pd.DataFrame(roll_dict, index=series.index)


def build_tabular_feature_dataset(
    ds: xr.Dataset,
    train_climatology: xr.DataArray | None = None,
    era5_ds: xr.Dataset | None = None,
    threshold: float = 50.0,
    lead_days: int = 1,
    feature_config: FeatureConfig | None = None,
    var_name: str = "rain",
    monsoon_months: list[int] | None = [6, 7, 8, 9],
    target_within_season_only: bool = True,
) -> pd.DataFrame:
    """Transform 3D spatio-temporal xarray dataset into tabular ML features and aligned targets.

    Causally Strict Design:
    - Features (lags, rolling windows) are computed per continuous year block to prevent cross-year leakage.
    - Each row records exact 'issue_date' and 'target_valid_date'.
    - If target_within_season_only is True, samples are filtered so BOTH issue_date and
      target_valid_date strictly reside within the monsoon season (e.g. JJAS), completely
      preventing season boundary jumps (e.g. Sept 30 predicting June 1 of next year).
    """
    cfg = feature_config or FeatureConfig()

    # Compute spatial neighborhood features if requested
    if cfg.include_spatial_neighbors:
        spat_mean_da, spat_max_da = compute_spatial_neighborhood_features(ds, var_name=var_name)
    else:
        spat_mean_da, spat_max_da = None, None

    records = []
    times = pd.to_datetime(ds["time"].values)
    lats = ds["lat"].values
    lons = ds["lon"].values

    # Pre-check available ERA5 variables
    era5_vars = []
    if era5_ds is not None:
        era5_vars = [v for v in ["msl", "tcwv", "u850", "v850", "z500"] if v in era5_ds]

    # Process each grid cell across time
    for lat_val in lats:
        for lon_val in lons:
            # Extract 1D time series for this grid cell
            rain_1d = pd.Series(ds[var_name].sel(lat=lat_val, lon=lon_val).values, index=times)

            # Skip cells with all NaN (ocean/unmonitored)
            if rain_1d.isna().all():
                continue

            df_cell = pd.DataFrame(index=times)
            df_cell["time"] = times
            df_cell["lat"] = lat_val
            df_cell["lon"] = lon_val
            df_cell["year"] = times.year
            df_cell["month"] = times.month
            df_cell["doy"] = times.dayofyear

            # Explicit issue and target valid calendar timestamps
            df_cell["issue_date"] = times
            df_cell["target_valid_date"] = times + pd.Timedelta(days=lead_days)

            # Current day rainfall at issue time t
            df_cell["rain_current"] = rain_1d.values

            # Grouping series by calendar year strictly prevents inter-annual/cross-season bleeding
            group_year = pd.Series(times.year, index=times)

            # Lagged features
            df_lags = compute_lagged_features(rain_1d, lags=cfg.lags, group_col=group_year)
            df_cell = pd.concat([df_cell, df_lags], axis=1)

            # Rolling features
            df_rolls = compute_rolling_features(
                rain_1d,
                rolling_sums=cfg.rolling_sums,
                rolling_max=cfg.rolling_max,
                rolling_mean=cfg.rolling_mean,
                group_col=group_year,
            )
            df_cell = pd.concat([df_cell, df_rolls], axis=1)

            # Spatial neighborhood features
            if spat_mean_da is not None and spat_max_da is not None:
                df_cell["rain_spatial_mean3x3"] = spat_mean_da.sel(lat=lat_val, lon=lon_val).values
                df_cell["rain_spatial_max3x3"] = spat_max_da.sel(lat=lat_val, lon=lon_val).values

            # Climatology feature (from training climatology)
            if cfg.include_climatology and train_climatology is not None:
                clim_cell = train_climatology.sel(lat=lat_val, lon=lon_val)
                clim_map = {
                    int(d): float(v)
                    for d, v in zip(clim_cell["dayofyear"].values, clim_cell.values)
                }
                df_cell["climatology_mean"] = df_cell["doy"].map(clim_map)
            elif cfg.include_climatology:
                df_cell["climatology_mean"] = 0.0

            # Cyclical calendar features
            if cfg.include_cyclical_calendar:
                df_cell["doy_sin"] = np.sin(2.0 * np.pi * df_cell["doy"] / 365.25)
                df_cell["doy_cos"] = np.cos(2.0 * np.pi * df_cell["doy"] / 365.25)

            # Atmospheric reanalysis predictors (ERA5 mode)
            if era5_vars and era5_ds is not None:
                for var in era5_vars:
                    era5_val = era5_ds[var].sel(lat=lat_val, lon=lon_val, method="nearest").values
                    df_cell[f"era5_{var}"] = era5_val
                if "u850" in era5_vars and "v850" in era5_vars:
                    u_arr = df_cell["era5_u850"].values
                    v_arr = df_cell["era5_v850"].values
                    df_cell["era5_wind_speed850"] = np.sqrt(u_arr**2 + v_arr**2)

            # Next-day Target strictly shifted within each year block
            # (Year boundary shift ensures the end of season gets NaN target, NEVER next year's June 1)
            target_amount = rain_1d.groupby(group_year).shift(-lead_days)
            df_cell["target_rainfall"] = target_amount.values
            df_cell["target_event"] = np.where(
                target_amount.isna(), np.nan, (target_amount >= threshold).astype(float)
            )

            # Filter to valid season issue dates and valid targets within season
            if monsoon_months is not None:
                if target_within_season_only:
                    valid_mask = (
                        df_cell["issue_date"].dt.month.isin(monsoon_months)
                        & df_cell["target_valid_date"].dt.month.isin(monsoon_months)
                    )
                else:
                    valid_mask = df_cell["issue_date"].dt.month.isin(monsoon_months)
                df_cell = df_cell[valid_mask].copy()

            records.append(df_cell)

    if not records:
        raise ValueError("No valid grid cell records were extracted from the dataset.")

    full_df = pd.concat(records, axis=0, ignore_index=True)

    # Drop rows where target is NaN (e.g. end of season or missing)
    full_df = full_df.dropna(subset=["target_rainfall", "target_event"]).reset_index(drop=True)

    # Fill early lag boundary NaNs with climatology mean or 0
    predictor_cols = [
        c
        for c in full_df.columns
        if c
        not in [
            "time",
            "issue_date",
            "target_valid_date",
            "lat",
            "lon",
            "year",
            "month",
            "doy",
            "target_rainfall",
            "target_event",
        ]
    ]
    for col in predictor_cols:
        full_df[col] = full_df[col].fillna(0.0)

    logger.info(
        f"Engineered tabular dataset: {len(full_df)} samples, {len(predictor_cols)} features."
    )
    return full_df
