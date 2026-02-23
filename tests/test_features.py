"""Unit tests for feature engineering, lags, and rolling windows."""

import numpy as np
import pandas as pd

from monsooncast.features.engineering import (
    compute_lagged_features,
    compute_rolling_features,
)


def test_lagged_features_causality():
    """Verify lagged features shift backward properly without looking into future."""
    dates = pd.date_range("2020-06-01", periods=10, freq="D")
    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    series = pd.Series(vals, index=dates)

    df_lags = compute_lagged_features(series, lags=[1, 2, 3])

    # lag 1 at index 1 (June 2) should be June 1 value (10.0)
    assert np.isnan(df_lags["rain_lag1"].iloc[0])
    assert df_lags["rain_lag1"].iloc[1] == 10.0
    assert df_lags["rain_lag1"].iloc[2] == 20.0

    # lag 2 at index 2 (June 3) should be June 1 value (10.0)
    assert np.isnan(df_lags["rain_lag2"].iloc[1])
    assert df_lags["rain_lag2"].iloc[2] == 10.0


def test_rolling_features_values():
    """Verify backward rolling sum and mean calculations."""
    dates = pd.date_range("2020-06-01", periods=5, freq="D")
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    series = pd.Series(vals, index=dates)

    df_roll = compute_rolling_features(series, rolling_sums=[3], rolling_max=[3], rolling_mean=[3])

    # 3-day sum at index 2 (10 + 20 + 30 = 60)
    assert df_roll["rain_roll_sum3"].iloc[2] == 60.0
    # 3-day mean at index 2 (60 / 3 = 20)
    assert df_roll["rain_roll_mean3"].iloc[2] == 20.0
    # 3-day max at index 2 is 30
    assert df_roll["rain_roll_max3"].iloc[2] == 30.0
