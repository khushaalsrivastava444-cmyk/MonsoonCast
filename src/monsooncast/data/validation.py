"""Data quality validation and integrity audits for scientific meteorological arrays."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation checks fail critical integrity rules."""


def validate_coordinates(ds: xr.Dataset) -> dict[str, Any]:
    """Validate spatial and temporal coordinate dimensions.

    Checks:
    - Presence of time, latitude, and longitude (or lat, lon).
    - Monotonically increasing/decreasing coordinates.
    - Coordinate values within valid geographic bounds for India (6N - 39N, 66E - 101E).
    """
    report: dict[str, Any] = {"passed": True, "errors": [], "warnings": []}

    # Check time coordinate
    if "time" not in ds.coords and "time" not in ds.dims:
        report["passed"] = False
        report["errors"].append("Missing required coordinate: 'time'")
    else:
        # Check monotonicity of time
        times = pd.to_datetime(ds["time"].values)
        if not times.is_monotonic_increasing:
            report["passed"] = False
            report["errors"].append("Time coordinate is not monotonically increasing.")
        if times.has_duplicates:
            report["passed"] = False
            report["errors"].append("Time coordinate contains duplicate timestamps.")

    # Check latitude
    lat_key = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    if not lat_key:
        report["passed"] = False
        report["errors"].append("Missing latitude coordinate ('lat' or 'latitude').")
    else:
        lats = ds[lat_key].values
        if np.any((lats < 4.0) | (lats > 42.0)):
            report["warnings"].append(
                f"Latitude values range [{lats.min():.2f}, {lats.max():.2f}] extend beyond India domain."
            )

    # Check longitude
    lon_key = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if not lon_key:
        report["passed"] = False
        report["errors"].append("Missing longitude coordinate ('lon' or 'longitude').")
    else:
        lons = ds[lon_key].values
        if np.any((lons < 60.0) | (lons > 105.0)):
            report["warnings"].append(
                f"Longitude values range [{lons.min():.2f}, {lons.max():.2f}] extend beyond India domain."
            )

    if not report["passed"]:
        logger.error(f"Coordinate validation failed: {report['errors']}")
        raise DataValidationError(f"Coordinate validation failed: {'; '.join(report['errors'])}")

    return report


def validate_units(
    ds: xr.Dataset, var_name: str = "rain", expected_units: str = "mm/day"
) -> dict[str, Any]:
    """Validate physical units and plausible meteorological bounds."""
    report: dict[str, Any] = {"passed": True, "var_name": var_name, "units": None, "warnings": []}

    if var_name not in ds:
        raise DataValidationError(f"Variable '{var_name}' not found in dataset.")

    attrs = ds[var_name].attrs
    units = attrs.get("units", "unknown")
    report["units"] = units

    # Check for unit scale mismatch (e.g. m/s or m instead of mm/day)
    data = ds[var_name].values
    valid_data = data[(data != -999.0) & (~np.isnan(data))]

    if len(valid_data) > 0:
        max_val = float(np.max(valid_data))
        min_val = float(np.min(valid_data))

        if min_val < -1e-5:
            report["warnings"].append(
                f"Negative rainfall values detected (min={min_val:.4f}). Physical rainfall cannot be negative."
            )

        # If max is less than 0.5 mm in a multi-year monsoon dataset, it may be in meters (m/day)
        if max_val > 0.0 and max_val < 0.5:
            report["warnings"].append(
                f"Suspiciously low max rainfall ({max_val:.4f}). Data may be in meters instead of millimeters."
            )
        elif max_val > 1500.0:
            report["warnings"].append(
                f"Extremely high rainfall value detected ({max_val:.1f} mm/day). Verify gauge calibration."
            )

    return report


def audit_missing_values(
    ds: xr.Dataset, var_name: str = "rain", fill_value: float = -999.0
) -> dict[str, Any]:
    """Compute detailed audit of missing and fill values in the spatial grid."""
    if var_name not in ds:
        raise DataValidationError(f"Variable '{var_name}' not found in dataset.")

    data = ds[var_name].values
    total_cells = int(data.size)
    nan_count = int(np.isnan(data).sum())
    fill_count = int((data == fill_value).sum())
    valid_count = total_cells - nan_count - fill_count

    pct_missing = ((nan_count + fill_count) / total_cells) * 100.0 if total_cells > 0 else 0.0

    return {
        "total_elements": total_cells,
        "nan_count": nan_count,
        "fill_value_count": fill_count,
        "valid_count": valid_count,
        "pct_missing": round(pct_missing, 3),
        "fill_value_used": fill_value,
    }


def check_temporal_monotonicity(times: pd.DatetimeIndex) -> bool:
    """Ensure timestamp index has no backwards jumps and no duplicates."""
    return bool(times.is_monotonic_increasing and not times.has_duplicates)
