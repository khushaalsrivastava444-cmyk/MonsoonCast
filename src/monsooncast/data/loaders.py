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

    # Generate daily dates across requested years including pre-monsoon (May) and post-monsoon (October) buffers
    # This guarantees uninterrupted daily continuity for lag calculations on June 1 and target alignment on September 30
    date_list = []
    for year in range(start_year, end_year + 1):
        dates = pd.date_range(start=f"{year}-05-15", end=f"{year}-10-15", freq="D")
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


def generate_synthetic_era5_dataset(
    output_path: str | Path | None = None,
    rainfall_ds: xr.Dataset | None = None,
    start_year: int = 2010,
    end_year: int = 2023,
    lat_bounds: tuple[float, float] = (18.0, 25.0),
    lon_bounds: tuple[float, float] = (73.0, 82.0),
    resolution: float = 0.5,
    seed: int = 42,
) -> xr.Dataset:
    """Generate physically plausible synthetic atmospheric reanalysis fields.

    Coupled dynamics:
    - High TCWV (55-70 kg/m2) preceding heavy precipitation.
    - Low-pressure anomalies (monsoon depression trough) down to 995 hPa.
    - Strong southwesterly low-level jet (u850 ~ 12-25 m/s, v850 ~ 5-15 m/s).
    - Lowered geopotential height at 500 hPa (z500 ~ 5820-5880 gpm).
    """
    logger.info("Generating synthetic atmospheric reanalysis (ERA5) test fixture...")
    rng = np.random.default_rng(seed)

    if rainfall_ds is not None:
        times = pd.DatetimeIndex(rainfall_ds["time"].values)
        lats = rainfall_ds["lat"].values
        lons = rainfall_ds["lon"].values
        rain_vals = rainfall_ds["rain"].values  # (time, lat, lon)
    else:
        date_list = []
        for year in range(start_year, end_year + 1):
            dates = pd.date_range(start=f"{year}-05-15", end=f"{year}-10-15", freq="D")
            date_list.extend(dates)
        times = pd.DatetimeIndex(date_list)
        lats = np.arange(lat_bounds[0], lat_bounds[1] + resolution, resolution)
        lons = np.arange(lon_bounds[0], lon_bounds[1] + resolution, resolution)
        rain_vals = np.zeros((len(times), len(lats), len(lons)))

    n_time, n_lat, n_lon = len(times), len(lats), len(lons)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Base monsoon climate state
    # MSLP: ~1004 hPa background, decreasing towards north (monsoon trough at ~24N)
    base_msl = 1008.0 - (lat_grid - 18.0) * 0.8  # (lat, lon)
    # TCWV: ~45 kg/m2 background, higher towards south and coasts
    base_tcwv = 48.0 + (25.0 - lat_grid) * 0.5 - (lon_grid - 73.0) * 0.4
    # U850: Westerly monsoon jet (~12 m/s)
    base_u850 = 12.0 + (lat_grid - 18.0) * 0.3
    # V850: Southerly component (~6 m/s)
    base_v850 = 6.0 - (lat_grid - 18.0) * 0.2
    # Z500: ~5860 gpm
    base_z500 = 5860.0 + (lat_grid - 18.0) * 2.0

    msl_arrays = []
    tcwv_arrays = []
    u850_arrays = []
    v850_arrays = []
    z500_arrays = []

    for t in range(n_time):
        day_rain = rain_vals[t] if rain_vals is not None else np.zeros((n_lat, n_lon))
        # Atmospheric depression anomaly linked with rainfall intensity
        rain_norm = np.clip(day_rain / 50.0, 0.0, 2.5)

        # MSLP anomaly: drop by up to 10 hPa in heavy rain regions
        msl_t = base_msl - 6.0 * rain_norm + rng.normal(0, 0.8, size=(n_lat, n_lon))
        # TCWV: surge by up to 18 kg/m2 during moist convection
        tcwv_t = base_tcwv + 12.0 * rain_norm + rng.normal(0, 1.5, size=(n_lat, n_lon))
        tcwv_t = np.clip(tcwv_t, 30.0, 75.0)
        # Low-level jet acceleration: U850 increases by up to 8 m/s
        u850_t = base_u850 + 5.0 * rain_norm + rng.normal(0, 1.2, size=(n_lat, n_lon))
        v850_t = base_v850 + 3.0 * rain_norm + rng.normal(0, 1.0, size=(n_lat, n_lon))
        # Mid-level geopotential height dip
        z500_t = base_z500 - 20.0 * rain_norm + rng.normal(0, 3.0, size=(n_lat, n_lon))

        msl_arrays.append(msl_t.astype(np.float32))
        tcwv_arrays.append(tcwv_t.astype(np.float32))
        u850_arrays.append(u850_t.astype(np.float32))
        v850_arrays.append(v850_t.astype(np.float32))
        z500_arrays.append(z500_t.astype(np.float32))

    ds_era5 = xr.Dataset(
        data_vars={
            "msl": (["time", "lat", "lon"], np.stack(msl_arrays, axis=0), {"units": "hPa", "long_name": "Mean Sea Level Pressure"}),
            "tcwv": (["time", "lat", "lon"], np.stack(tcwv_arrays, axis=0), {"units": "kg m**-2", "long_name": "Total Column Water Vapour"}),
            "u850": (["time", "lat", "lon"], np.stack(u850_arrays, axis=0), {"units": "m s**-1", "long_name": "Zonal Wind at 850 hPa"}),
            "v850": (["time", "lat", "lon"], np.stack(v850_arrays, axis=0), {"units": "m s**-1", "long_name": "Meridional Wind at 850 hPa"}),
            "z500": (["time", "lat", "lon"], np.stack(z500_arrays, axis=0), {"units": "gpm", "long_name": "Geopotential Height at 500 hPa"}),
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "title": "MonsoonCast Synthetic Coupled ERA5 Atmospheric Reanalysis",
            "is_synthetic": 1,
            "synthetic_disclaimer": "SYNTHETIC FIXTURE FOR PIPELINE TESTING ONLY. DO NOT USE AS REAL REANALYSIS.",
            "source": "MonsoonCast Reanalysis Generator",
        },
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds_era5.to_netcdf(out)
        logger.info(f"Saved synthetic ERA5 dataset to: {out}")

    return ds_era5


class ERA5Adapter:
    """Adapter for ERA5 atmospheric reanalysis single-level and pressure-level NetCDF files."""

    def __init__(self, data_dir: str | Path = "data/raw/era5"):
        self.data_dir = Path(data_dir)

    def is_available(self) -> bool:
        """Check if ERA5 NetCDF files exist locally."""
        return self.data_dir.is_dir() and any(self.data_dir.glob("*.nc"))

    def load_variables(
        self,
        variables: list[str] | None = None,
        generate_fixture_if_missing: bool = True,
        rainfall_ds: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Load ERA5 atmospheric variables aligned with target spatial domain."""
        if self.is_available():
            files = list(self.data_dir.glob("*.nc"))
            logger.info(f"Loading {len(files)} local ERA5 NetCDF files from {self.data_dir}...")
            ds = xr.open_mfdataset(files)
            return standardize_dataset_coords(ds)
        elif generate_fixture_if_missing:
            logger.warning(
                f"ERA5 data not found in {self.data_dir}. Generating synthetic coupled atmospheric reanalysis fixture..."
            )
            fixture_path = self.data_dir / "era5_synthetic_coupled.nc"
            return generate_synthetic_era5_dataset(output_path=fixture_path, rainfall_ds=rainfall_ds)
        else:
            raise FileNotFoundError(
                f"ERA5 reanalysis files not found in {self.data_dir}. "
                "Download ERA5 single/pressure levels from Copernicus CDS as documented in data/README.md."
            )


class ForecastEnsembleAdapter:
    """Adapter for Numerical Weather Prediction (NWP) Ensemble Member Forecasts (e.g., TIGGE/ECMWF).

    Enforces strict issue-time, lead-time, and valid-time causality:
    valid_time = initialization_time + lead_time
    """

    def __init__(self, data_dir: str | Path = "data/raw/tigge"):
        self.data_dir = Path(data_dir)

    def is_available(self) -> bool:
        return self.data_dir.is_dir() and any(self.data_dir.glob("*.nc"))

    def generate_synthetic_forecasts(
        self,
        obs_ds: xr.Dataset,
        lead_time_hours: int = 24,
        n_members: int = 10,
        output_path: str | Path | None = None,
        seed: int = 42,
    ) -> xr.Dataset:
        """Generate synthetic NWP ensemble forecasts with issue-time and lead-time alignment.

        Simulates ensemble member forecasts around observed precipitation with
        realistic chaotic dispersion:
        f_m(t_valid) = R(t_valid) + bias + noise_m
        """
        logger.info(f"Generating synthetic NWP {n_members}-member forecast ensemble (lead={lead_time_hours}h)...")
        rng = np.random.default_rng(seed)

        valid_times = pd.DatetimeIndex(obs_ds["time"].values)
        init_times = valid_times - pd.Timedelta(hours=lead_time_hours)
        lats = obs_ds["lat"].values
        lons = obs_ds["lon"].values
        rain_obs = obs_ds["rain"].values  # (time, lat, lon)

        members = [f"mem_{m:02d}" for m in range(1, n_members + 1)]
        forecast_members = []

        for m in range(n_members):
            # Member perturbation (lead-time dependent uncertainty)
            spread_factor = 0.25 * (lead_time_hours / 24.0)
            member_noise = rng.normal(0.0, 4.0 + spread_factor * rain_obs, size=rain_obs.shape)
            member_fc = np.clip(rain_obs + member_noise, 0.0, None)
            forecast_members.append(member_fc.astype(np.float32))

        fc_tensor = np.stack(forecast_members, axis=1)  # (time, member, lat, lon)

        ds_fc = xr.Dataset(
            data_vars={
                "tp": (["time", "member", "lat", "lon"], fc_tensor, {"units": "mm", "long_name": "Total Precipitation Forecast"}),
                "ens_mean": (["time", "lat", "lon"], np.mean(fc_tensor, axis=1), {"units": "mm"}),
                "ens_spread": (["time", "lat", "lon"], np.std(fc_tensor, axis=1), {"units": "mm"}),
            },
            coords={
                "time": valid_times,
                "initialization_time": ("time", init_times),
                "member": members,
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "title": f"MonsoonCast Synthetic NWP Forecast Ensemble ({lead_time_hours}h lead)",
                "lead_time_hours": lead_time_hours,
                "n_members": n_members,
                "is_synthetic": 1,
            },
        )

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            ds_fc.to_netcdf(out)
            logger.info(f"Saved synthetic forecast ensemble to: {out}")

        return ds_fc
