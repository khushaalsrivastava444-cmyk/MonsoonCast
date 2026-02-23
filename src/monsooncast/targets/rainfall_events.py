"""Target label generation for extreme monsoon rainfall events."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def create_binary_event_labels(rainfall_array: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """Create binary indicator for extreme rainfall: 1 if rain >= threshold, else 0.

    Handles NaNs safely: returns NaN where rainfall is unmeasured.
    """
    labels = np.where(np.isnan(rainfall_array), np.nan, (rainfall_array >= threshold).astype(float))
    return labels


def compute_empirical_percentile_thresholds(
    train_ds: xr.Dataset,
    var_name: str = "rain",
    percentile: float = 95.0,
    wet_day_threshold: float = 0.1,
) -> xr.DataArray:
    """Compute local grid-cell percentile thresholds strictly from training data.

    Computed only over 'wet days' (rain >= wet_day_threshold) to prevent
    zero-inflation from skewing high percentiles into dry days.
    """
    rain = train_ds[var_name]
    # Mask dry days
    wet_rain = rain.where(rain >= wet_day_threshold)

    # Compute percentile along time dimension for each (lat, lon)
    pct_thresh = wet_rain.quantile(percentile / 100.0, dim="time", skipna=True)
    pct_thresh.attrs["description"] = (
        f"Training {percentile}th percentile on wet days (>= {wet_day_threshold} mm)"
    )
    pct_thresh.attrs["units"] = "mm/day"
    return pct_thresh


def build_aligned_target_series(
    rainfall_series: pd.Series, lead_days: int = 1, threshold: float = 50.0
) -> tuple[pd.Series, pd.Series]:
    """Given daily rainfall series at day t, create next-day target:
    - Target amount: R(t + lead_days)
    - Target binary label: Y(t + lead_days) >= threshold

    Note: Output series are indexed by the prediction issue time t!
    """
    # Shift backwards so index t gets value at t + lead_days
    target_amount = rainfall_series.shift(-lead_days)
    target_label = (target_amount >= threshold).astype(float)
    # Mask where target amount is NaN
    target_label = target_label.where(target_amount.notna(), np.nan)
    return target_amount, target_label
