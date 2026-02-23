"""Unit tests for data validation, coordinate standardization, and synthetic generation."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monsooncast.data.loaders import generate_synthetic_monsoon_dataset
from monsooncast.data.validation import (
    DataValidationError,
    audit_missing_values,
    validate_coordinates,
)


def test_synthetic_monsoon_generator():
    """Verify synthetic dataset generator produces valid, non-empty xarray dataset."""
    ds = generate_synthetic_monsoon_dataset(
        start_year=2020,
        end_year=2021,
        lat_bounds=(18.0, 20.0),
        lon_bounds=(73.0, 75.0),
        resolution=1.0,
        seed=123,
    )
    assert isinstance(ds, xr.Dataset)
    assert "rain" in ds
    assert ds.attrs.get("is_synthetic") in (1, True, "true")
    assert ds.sizes["lat"] > 0
    assert ds.sizes["lon"] > 0
    # 154 days per year: May 15 to Oct 15 buffer for uninterrupted lag/lead calculations
    assert ds.sizes["time"] == 154 * 2
    # Verify May buffer and October buffer presence
    times = pd.to_datetime(ds.time.values)
    assert 5 in times.month
    assert 10 in times.month


def test_validate_coordinates_valid():
    """Verify coordinate validator succeeds on valid spatial bounds."""
    lats = np.array([18.0, 19.0, 20.0])
    lons = np.array([73.0, 74.0, 75.0])
    times = pd.date_range("2020-06-01", periods=10, freq="D")
    data = np.ones((10, 3, 3))

    ds = xr.Dataset(
        data_vars={"rain": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    report = validate_coordinates(ds)
    assert report["passed"] is True
    assert len(report["errors"]) == 0


def test_validate_coordinates_missing_lat_raises():
    """Verify validator raises DataValidationError when latitude is missing."""
    times = pd.date_range("2020-06-01", periods=5, freq="D")
    ds = xr.Dataset(data_vars={"rain": (["time"], np.ones(5))}, coords={"time": times})
    with pytest.raises(DataValidationError):
        validate_coordinates(ds)


def test_audit_missing_values():
    """Verify missing value auditing detects fill values and NaNs accurately."""
    data = np.array([[[10.0, -999.0], [np.nan, 25.0]]])
    ds = xr.Dataset(
        data_vars={"rain": (["time", "lat", "lon"], data)},
        coords={"time": [0], "lat": [0, 1], "lon": [0, 1]},
    )
    audit = audit_missing_values(ds, var_name="rain", fill_value=-999.0)
    assert audit["total_elements"] == 4
    assert audit["fill_value_count"] == 1
    assert audit["nan_count"] == 1
    assert audit["valid_count"] == 2
    assert audit["pct_missing"] == 50.0
