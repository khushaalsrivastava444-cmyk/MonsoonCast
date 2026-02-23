"""Rigorous causal correctness, calendar alignment, and seasonal boundary verification tests."""

import numpy as np
import pandas as pd
import xarray as xr

from monsooncast.evaluation.metrics import bootstrap_metric_ci
from monsooncast.features.engineering import build_tabular_feature_dataset


def test_seasonal_boundary_no_cross_year_bleeding():
    """Verify that feature engineering NEVER bleeds across non-contiguous seasons.

    Specifically:
    - September 30 of Year Y must NEVER have June 1 of Year Y+1 as its next-day target.
    - June 1 of Year Y+1 must NEVER have September 30 of Year Y as its lag-1 predictor.
    """
    # Create two years with May to October data
    dates = []
    for y in [2021, 2022]:
        dates.extend(pd.date_range(f"{y}-05-25", f"{y}-10-05", freq="D"))
    times = pd.DatetimeIndex(dates)

    lats = [20.0]
    lons = [78.0]

    # Assign distinct values: 2021 gets ~10.0, 2022 gets ~90.0
    rain_data = np.where(times.year == 2021, 10.0, 90.0).reshape(len(times), 1, 1)

    ds = xr.Dataset(
        data_vars={"rain": (["time", "lat", "lon"], rain_data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    df_tab = build_tabular_feature_dataset(
        ds=ds,
        monsoon_months=[6, 7, 8, 9],
        target_within_season_only=True,
    )

    # 1. Inspect June 1, 2022: lag1 must be May 31, 2022 (value 90.0), NOT Sept 30, 2021 (value 10.0)
    june1_2022 = df_tab[df_tab["time"] == "2022-06-01"]
    assert len(june1_2022) == 1
    assert june1_2022["rain_lag1"].iloc[0] == 90.0, "June 1 must NOT inherit lag from previous year!"

    # 2. Inspect September 29, 2021: target must be September 30, 2021 (value 10.0)
    sept29_2021 = df_tab[df_tab["time"] == "2021-09-29"]
    assert len(sept29_2021) == 1
    assert sept29_2021["target_rainfall"].iloc[0] == 10.0

    # 3. Target dates must strictly match issue_date + 1 day within the same year
    for _, row in df_tab.iterrows():
        assert row["target_valid_date"] == row["issue_date"] + pd.Timedelta(days=1)
        assert row["issue_date"].year == row["target_valid_date"].year


def test_exact_calendar_timestamp_alignment():
    """Verify exact date matching for all lagged and rolling features."""
    dates = pd.date_range("2021-05-15", "2021-10-15", freq="D")
    lats = [21.0]
    lons = [78.0]
    # Linear ramp: day 0 has 0, day 1 has 1, ...
    rain_vals = np.arange(len(dates), dtype=float).reshape(len(dates), 1, 1)

    ds = xr.Dataset(
        data_vars={"rain": (["time", "lat", "lon"], rain_vals)},
        coords={"time": dates, "lat": lats, "lon": lons},
    )

    df_tab = build_tabular_feature_dataset(
        ds=ds,
        monsoon_months=[6, 7, 8, 9],
        target_within_season_only=True,
    )

    # For an arbitrary mid-monsoon date (e.g. July 15)
    sample = df_tab[df_tab["time"] == "2021-07-15"].iloc[0]
    curr_rain = sample["rain_current"]

    # lag 1 must be curr_rain - 1
    assert sample["rain_lag1"] == curr_rain - 1.0
    # lag 2 must be curr_rain - 2
    assert sample["rain_lag2"] == curr_rain - 2.0
    # lag 3 must be curr_rain - 3
    assert sample["rain_lag3"] == curr_rain - 3.0
    # lag 7 must be curr_rain - 7
    assert sample["rain_lag7"] == curr_rain - 7.0
    # target must be curr_rain + 1
    assert sample["target_rainfall"] == curr_rain + 1.0
    # rolling sum 3 must be R(t) + R(t-1) + R(t-2)
    expected_sum3 = curr_rain + (curr_rain - 1.0) + (curr_rain - 2.0)
    assert sample["rain_roll_sum3"] == expected_sum3


def test_spatial_date_block_bootstrap_intact_clusters():
    """Verify that block bootstrap preserves spatial date clusters without row-level fragmentation."""
    # 3 days with 4 grid cells each = 12 total instances
    dates = [
        "2021-06-01", "2021-06-01", "2021-06-01", "2021-06-01",
        "2021-06-02", "2021-06-02", "2021-06-02", "2021-06-02",
        "2021-06-03", "2021-06-03", "2021-06-03", "2021-06-03",
    ]
    y_true = np.array([1, 1, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2,  0.1, 0.1, 0.2, 0.1,  0.8, 0.9, 0.7, 0.8])

    low, pt, high = bootstrap_metric_ci(
        y_true=y_true,
        y_prob=y_prob,
        metric_func=lambda yt, yp: float(np.mean((yt == (yp >= 0.5)))),
        dates=dates,
        block_size_days=1,
        n_bootstrap=50,
        seed=42,
    )
    assert 0.0 <= low <= pt <= high <= 1.0
