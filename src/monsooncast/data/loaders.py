"""Data loaders and adapters for IMD gridded rainfall, ERA5 reanalysis, and NWP forecasts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from monsooncast.data.validation import validate_coordinates, validate_units

logger = logging.getLogger(__name__)


def standardize_dataset_coords(ds: xr.Dataset) -> xr.Dataset:
    """Standardize coordinate names to ('time', 'lat', 'lon')."""
    coord_map = {}
    for c in ds.coords:
        c_lower = c.lower()
        if c_lower in ("latitude", "lats"):
            coord_map[c] = "lat"
        elif c_lower in ("longitude", "longitudes", "lons"):
            coord_map[c] = "lon"
        elif c_lower in ("times", "date", "dates"):
            coord_map[c] = "time"

    if coord_map:
        ds = ds.rename(coord_map)

    # Ensure latitude is sorted ascending
    if "lat" in ds.coords:
        if ds["lat"].values[0] > ds["lat"].values[-1]:
            ds = ds.reindex(lat=list(reversed(ds["lat"].values)))

    # Ensure longitude is sorted ascending
    if "lon" in ds.coords:
        if ds["lon"].values[0] > ds["lon"].values[-1]:
            ds = ds.reindex(lon=list(reversed(ds["lon"].values)))

    # Ensure time is datetime64[ns]
    if "time" in ds.coords:
        if not np.issubdtype(ds["time"].dtype, np.datetime64):
            ds["time"] = pd.to_datetime(ds["time"].values)

    return ds


def load_imd_netcdf(
    filepath: str | Path, var_name: str = "rain", fill_value: float = -999.0
) -> xr.Dataset:
    """Load and validate an official IMD daily gridded rainfall NetCDF file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"IMD NetCDF file not found: {path}")

    logger.info(f"Opening IMD NetCDF file: {path.name}")
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
    except Exception:
        # Fallback to h5netcdf or default engine
        ds = xr.open_dataset(path)

    ds = standardize_dataset_coords(ds)

    # Auto-detect rainfall variable if name differs
    if var_name not in ds:
        candidates = [
            v
            for v in ds.data_vars
            if "rain" in v.lower() or "precip" in v.lower() or "rf" in v.lower()
        ]
        if candidates:
            logger.warning(
                f"Variable '{var_name}' not in dataset; auto-detected candidate '{candidates[0]}'"
            )
            ds = ds.rename({candidates[0]: var_name})
        else:
            raise KeyError(
                f"Variable '{var_name}' not found. Available data variables: {list(ds.data_vars)}"
            )

    # Coordinate validation
    validate_coordinates(ds)
    validate_units(ds, var_name=var_name)

    # Clean fill value to NaN in an interim data array
    rain_data = ds[var_name].values
    rain_clean = np.where(rain_data == fill_value, np.nan, rain_data)
    # Physical clip
    rain_clean = np.where(rain_clean < 0.0, 0.0, rain_clean)
    ds[var_name].values = rain_clean

    ds.attrs["provenance"] = f"IMD Daily Gridded Rainfall loaded from {path.name}"
    ds.attrs["is_synthetic"] = False
    return ds


def generate_synthetic_monsoon_dataset(
    output_path: str | Path | None = None,
    start_year: int = 2010,
    end_year: int = 2023,
    lat_bounds: tuple[float, float] = (18.0, 25.0),
    lon_bounds: tuple[float, float] = (73.0, 82.0),
    resolution: float = 0.5,
    seed: int = 42,
) -> xr.Dataset:
    """Generate a scientifically structured synthetic monsoon dataset for pipeline validation.

    CRITICAL POLICY:
    - This function is strictly used for software unit tests, integration verification,
      and CI testing when large observational archives are not yet downloaded.
    - It is explicitly marked with `is_synthetic = True` and must never be cited as real evidence.

    Simulates:
    - Summer monsoon seasonality (June-Sept peak in July/August).
    - Orographic rainfall gradient across Western Ghats (high rain on west, rain-shadow on east).
    - Zero-inflated Gamma distribution (many dry/light rain days, realistic extreme tail).
    """
    logger.info("Generating synthetic meteorological test fixture dataset...")
    rng = np.random.default_rng(seed)

    # Coordinate grids
    lats = np.arange(lat_bounds[0], lat_bounds[1] + resolution, resolution)
    lons = np.arange(lon_bounds[0], lon_bounds[1] + resolution, resolution)
    n_lat, n_lon = len(lats), len(lons)

    # Generate daily dates for June-September across requested years
    date_list = []
    for year in range(start_year, end_year + 1):
        dates = pd.date_range(start=f"{year}-06-01", end=f"{year}-09-30", freq="D")
        date_list.extend(dates)
    times = pd.DatetimeIndex(date_list)

    # 2D Spatial orographic field (higher towards West: lon 73)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    orographic_factor = np.exp(-(lon_grid - 73.0) / 4.0) + 0.3 * np.sin((lat_grid - 18.0) / 3.0)
    orographic_factor = np.clip(orographic_factor, 0.5, 3.5)

    # Daily temporal seasonality (peak around day 50 of monsoon, ~late July)
    rain_arrays = []
    for t in times:
        doy = t.dayofyear
        # Monsoon seasonal bell curve centered around DOY 205 (July 24)
        seasonal_intensity = np.exp(-((doy - 205) ** 2) / (2 * (30**2)))

        # Spatial rain field
        # Probability of wet day
        p_wet = 0.4 + 0.35 * seasonal_intensity
        wet_mask = rng.random((n_lat, n_lon)) < p_wet

        # Gamma distributed rainfall amounts on wet days
        gamma_shape = 1.2
        gamma_scale = (8.0 + 12.0 * seasonal_intensity) * orographic_factor
        amounts = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=(n_lat, n_lon))

        # Occasional synoptic monsoon depression causing extreme event (> 50 mm/day)
        if rng.random() < 0.08 * seasonal_intensity:
            depression_center_lat = rng.choice(lats)
            depression_center_lon = rng.choice(lons)
            dist_sq = (lat_grid - depression_center_lat) ** 2 + (
                lon_grid - depression_center_lon
            ) ** 2
            depression_rain = 75.0 * np.exp(-dist_sq / 4.0)
            amounts += depression_rain

        rain_day = np.where(wet_mask, amounts, 0.0)
        rain_day = np.round(rain_day, 2)
        rain_arrays.append(rain_day)

    rain_tensor = np.stack(rain_arrays, axis=0)  # (time, lat, lon)

    ds = xr.Dataset(
        data_vars={
            "rain": (
                ["time", "lat", "lon"],
                rain_tensor,
                {"units": "mm/day", "long_name": "Daily Rainfall"},
            )
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "title": "MonsoonCast Synthetic Testing NetCDF Dataset",
            "is_synthetic": 1,
            "synthetic_disclaimer": "FOR PIPELINE INTEGRATION TESTS ONLY. DO NOT USE AS REAL CLIMATE DATA.",
            "spatial_resolution_deg": float(resolution),
            "source": "MonsoonCast Synthetic Meteorological Generator",
        },
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out)
        logger.info(f"Saved synthetic dataset to: {out} ({ds['rain'].shape})")

    return ds


class ERA5Adapter:
    """Adapter for ERA5 atmospheric reanalysis single-level and pressure-level NetCDF files."""

    def __init__(self, data_dir: str | Path = "data/raw/era5"):
        self.data_dir = Path(data_dir)

    def is_available(self) -> bool:
        """Check if ERA5 NetCDF files exist locally."""
        return self.data_dir.is_dir() and any(self.data_dir.glob("*.nc"))

    def load_variables(self, variables: list[str], start_date: str, end_date: str) -> xr.Dataset:
        """Load ERA5 variables aligned with the target spatial-temporal grid."""
        if not self.is_available():
            raise FileNotFoundError(
                f"ERA5 reanalysis files not found in {self.data_dir}. "
                "Download ERA5 single/pressure levels from Copernicus CDS as documented in data/README.md."
            )
        # Placeholder for real ERA5 loader
        files = list(self.data_dir.glob("*.nc"))
        ds = xr.open_mfdataset(files)
        return standardize_dataset_coords(ds)


class ForecastEnsembleAdapter:
    """Adapter for Archived Numerical Weather Prediction (NWP) Ensemble Forecasts (e.g., TIGGE).

    Enforces strict issue-time, lead-time, and valid-time causality:
    valid_time = initialization_time + lead_time
    """

    def __init__(self, data_dir: str | Path = "data/raw/tigge"):
        self.data_dir = Path(data_dir)

    def is_available(self) -> bool:
        return self.data_dir.is_dir() and any(self.data_dir.glob("*.nc"))

    def load_forecasts(
        self, initialization_times: list[pd.Timestamp], lead_time_hours: int = 24
    ) -> xr.Dataset:
        """Load ensemble forecasts initialized strictly before target valid time."""
        if not self.is_available():
            raise FileNotFoundError(
                f"Archived NWP forecast data not found in {self.data_dir}. "
                "Archived TIGGE forecasts required for Mode B evaluation."
            )
        raise NotImplementedError("Mode B requires verified local TIGGE NetCDF files.")
