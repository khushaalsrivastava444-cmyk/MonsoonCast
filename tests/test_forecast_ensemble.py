"""Unit tests for Mode A (ERA5 Reanalysis) and Mode B (NWP Forecast Ensemble) Adapters."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monsooncast.data.loaders import (
    ERA5Adapter,
    ForecastEnsembleAdapter,
    generate_synthetic_era5_dataset,
    generate_synthetic_monsoon_dataset,
)


@pytest.fixture
def sample_rainfall_ds():
    """Create a minimal 2-year JJAS rainfall dataset for testing."""
    return generate_synthetic_monsoon_dataset(
        start_year=2021,
        end_year=2022,
        lat_bounds=(19.0, 21.0),
        lon_bounds=(74.0, 76.0),
        resolution=1.0,
        seed=42,
    )


def test_synthetic_era5_generator(sample_rainfall_ds):
    """Verify ERA5 reanalysis generator outputs coupled meteorological fields."""
    ds_era5 = generate_synthetic_era5_dataset(
        rainfall_ds=sample_rainfall_ds,
        seed=101,
    )
    assert isinstance(ds_era5, xr.Dataset)
    for var in ["msl", "tcwv", "u850", "v850", "z500"]:
        assert var in ds_era5, f"Missing expected variable {var}"

    # Verify physical sanity
    assert float(ds_era5["tcwv"].min()) >= 25.0, "TCWV should be realistic monsoon water vapor"
    assert float(ds_era5["tcwv"].max()) <= 85.0
    assert float(ds_era5["msl"].min()) >= 980.0
    assert float(ds_era5["msl"].max()) <= 1025.0

    # Verify coordinate alignment with rainfall dataset
    np.testing.assert_array_equal(ds_era5["lat"].values, sample_rainfall_ds["lat"].values)
    np.testing.assert_array_equal(ds_era5["lon"].values, sample_rainfall_ds["lon"].values)
    np.testing.assert_array_equal(ds_era5["time"].values, sample_rainfall_ds["time"].values)


def test_era5_adapter(tmp_path, sample_rainfall_ds):
    """Verify ERA5Adapter loads or generates synthetic coupled data on demand."""
    era5_dir = tmp_path / "era5"
    adapter = ERA5Adapter(data_dir=era5_dir)
    assert not adapter.is_available()

    # Load with automatic generation
    ds = adapter.load_variables(rainfall_ds=sample_rainfall_ds)
    assert isinstance(ds, xr.Dataset)
    assert "tcwv" in ds
    assert adapter.is_available()


def test_forecast_ensemble_adapter_causality(sample_rainfall_ds):
    """Verify Mode B NWP Forecast Ensemble enforces strict initialization/valid causality:

    valid_time = initialization_time + lead_time
    """
    adapter = ForecastEnsembleAdapter()
    lead_hours = 24
    n_members = 5

    ds_fc = adapter.generate_synthetic_forecasts(
        obs_ds=sample_rainfall_ds,
        lead_time_hours=lead_hours,
        n_members=n_members,
        seed=77,
    )

    assert isinstance(ds_fc, xr.Dataset)
    assert "tp" in ds_fc
    assert "ens_mean" in ds_fc
    assert "ens_spread" in ds_fc
    assert ds_fc.sizes["member"] == n_members

    # Check causality alignment
    valid_times = pd.DatetimeIndex(ds_fc["time"].values)
    init_times = pd.DatetimeIndex(ds_fc["initialization_time"].values)

    expected_init = valid_times - pd.Timedelta(hours=lead_hours)
    pd.testing.assert_index_equal(init_times, expected_init)

    # Check ensemble spread non-negativity
    assert (ds_fc["ens_spread"].values >= 0).all()
    # Check ensemble attributes
    assert ds_fc.attrs["lead_time_hours"] == lead_hours
    assert ds_fc.attrs["is_synthetic"] == 1
