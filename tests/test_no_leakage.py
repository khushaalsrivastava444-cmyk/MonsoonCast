"""Tests strictly enforcing absence of target leakage and temporal leakage."""

import numpy as np
import pandas as pd
import pytest

from monsooncast.evaluation.validation import (
    TargetLeakageError,
    chronological_split,
    verify_no_leakage,
)


def test_chronological_split_strictness():
    """Verify chronological split guarantees non-overlapping partitions by year."""
    dates = pd.date_range("2015-06-01", "2023-09-30", freq="D")
    df = pd.DataFrame(
        {
            "time": dates,
            "year": dates.year,
            "target_event": np.random.randint(0, 2, size=len(dates)),
            "feat1": np.random.randn(len(dates)),
        }
    )

    train_df, val_df, test_df = chronological_split(
        df,
        train_years=[2015, 2016, 2017, 2018],
        val_years=[2019, 2020],
        test_years=[2021, 2022, 2023],
    )

    assert set(train_df["year"].unique()) == {2015, 2016, 2017, 2018}
    assert set(val_df["year"].unique()) == {2019, 2020}
    assert set(test_df["year"].unique()) == {2021, 2022, 2023}

    assert train_df["time"].max() < val_df["time"].min()
    assert val_df["time"].max() < test_df["time"].min()


def test_verify_no_leakage_passes_on_valid_data():
    """Verify leak detector passes clean dataset."""
    train_dates = pd.date_range("2018-06-01", "2018-09-30")
    val_dates = pd.date_range("2019-06-01", "2019-09-30")
    test_dates = pd.date_range("2020-06-01", "2020-09-30")

    train_df = pd.DataFrame(
        {
            "time": train_dates,
            "feat": np.random.randn(len(train_dates)),
            "target_event": np.random.randint(0, 2, len(train_dates)),
        }
    )
    val_df = pd.DataFrame(
        {
            "time": val_dates,
            "feat": np.random.randn(len(val_dates)),
            "target_event": np.random.randint(0, 2, len(val_dates)),
        }
    )
    test_df = pd.DataFrame(
        {
            "time": test_dates,
            "feat": np.random.randn(len(test_dates)),
            "target_event": np.random.randint(0, 2, len(test_dates)),
        }
    )

    report = verify_no_leakage(
        train_df, val_df, test_df, feature_cols=["feat"], target_col="target_event"
    )
    assert report["passed"] is True


def test_verify_no_leakage_detects_target_in_features():
    """Verify leak detector raises error if target column is included in feature_cols."""
    train_dates = pd.date_range("2018-06-01", "2018-09-30")
    val_dates = pd.date_range("2019-06-01", "2019-09-30")
    test_dates = pd.date_range("2020-06-01", "2020-09-30")

    train_df = pd.DataFrame({"time": train_dates, "target_event": [0, 1] * 61})
    val_df = pd.DataFrame({"time": val_dates, "target_event": [0, 1] * 61})
    test_df = pd.DataFrame({"time": test_dates, "target_event": [0, 1] * 61})

    with pytest.raises(TargetLeakageError):
        verify_no_leakage(
            train_df, val_df, test_df, feature_cols=["target_event"], target_col="target_event"
        )
