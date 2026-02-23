"""Data preprocessing, spatial subsetting, and seasonal filtering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from monsooncast.config import SpatialBounds, TemporalScope

logger = logging.getLogger(__name__)


def subset_spatial(
    ds: xr.Dataset, bounds: SpatialBounds | tuple[float, float, float, float]
) -> xr.Dataset:
    """Subset xarray dataset to specified bounding box [lat_min, lat_max, lon_min, lon_max]."""
    if isinstance(bounds, SpatialBounds):
        lat_min, lat_max = bounds.lat_min, bounds.lat_max
        lon_min, lon_max = bounds.lon_min, bounds.lon_max
    else:
        lat_min, lat_max, lon_min, lon_max = bounds

    # Check whether lat is ascending or descending
    lats = ds["lat"].values
    if lats[0] <= lats[-1]:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)

    lons = ds["lon"].values
    if lons[0] <= lons[-1]:
        lon_slice = slice(lon_min, lon_max)
    else:
        lon_slice = slice(lon_max, lon_min)

    sub = ds.sel(lat=lat_slice, lon=lon_slice)
    if sub.sizes["lat"] == 0 or sub.sizes["lon"] == 0:
        raise ValueError(
            f"Spatial subset resulted in 0 points for bounds: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]"
        )

    logger.info(f"Spatial subset: lat={sub.sizes['lat']} points, lon={sub.sizes['lon']} points")
    return sub


def filter_monsoon_season(ds: xr.Dataset, months: list[int] = [6, 7, 8, 9]) -> xr.Dataset:
    """Filter dataset to Indian Summer Monsoon months (default June-September: [6, 7, 8, 9])."""
    # Extract month from time coordinate
    times = pd.to_datetime(ds["time"].values)
    mask = times.month.isin(months)

    if not np.any(mask):
        raise ValueError(f"No observations found in dataset matching monsoon months: {months}")

    filtered = ds.isel(time=mask)
    logger.info(
        f"Monsoon season filter (months={months}): {filtered.sizes['time']} daily timestamps retained."
    )
    return filtered


def align_and_preprocess_dataset(
    ds: xr.Dataset,
    spatial_bounds: SpatialBounds,
    temporal_scope: TemporalScope,
    var_name: str = "rain",
) -> xr.Dataset:
    """Perform coordinate standardization, spatial bounding, and seasonal filtering."""
    # Spatial subsetting
    ds_sub = subset_spatial(ds, spatial_bounds)

    # Monsoon season filtering
    ds_monsoon = filter_monsoon_season(ds_sub, temporal_scope.monsoon_months)

    # Ensure no negative rainfall
    rain = ds_monsoon[var_name].values
    rain = np.where(rain < 0.0, 0.0, rain)
    ds_monsoon[var_name].values = rain

    return ds_monsoon
