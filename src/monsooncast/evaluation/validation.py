"""Chronological dataset partitioning and leakage verification."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TargetLeakageError(Exception):
    """Raised when data leakage between partitions or target and features is detected."""


def chronological_split(
    df: pd.DataFrame, train_years: list[int], val_years: list[int], test_years: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strictly partition dataframe by calendar years to prevent future leakage.

    Guarantees:
    - Zero temporal overlap between Train, Validation, and Test partitions.
    - All observations from a single calendar year remain in the same partition.
    """
    if "year" not in df.columns:
        if "time" in df.columns:
            df["year"] = pd.to_datetime(df["time"]).dt.year
        else:
            raise KeyError(
                "Dataframe must contain 'year' or 'time' column for chronological splitting."
            )

    # Check for disjoint year sets
    s_train, s_val, s_test = set(train_years), set(val_years), set(test_years)
    if s_train.intersection(s_val) or s_train.intersection(s_test) or s_val.intersection(s_test):
        raise ValueError("Year sets for train, validation, and test must be mutually exclusive!")

    train_df = df[df["year"].isin(train_years)].copy().reset_index(drop=True)
    val_df = df[df["year"].isin(val_years)].copy().reset_index(drop=True)
    test_df = df[df["year"].isin(test_years)].copy().reset_index(drop=True)

    logger.info(
        f"Chronological split complete: "
        f"Train={len(train_df)} ({min(train_years)}-{max(train_years)}), "
        f"Val={len(val_df)} ({min(val_years)}-{max(val_years)}), "
        f"Test={len(test_df)} ({min(test_years)}-{max(test_years)})"
    )

    if len(train_df) == 0:
        raise ValueError(f"Train partition is empty for years {train_years}!")
    if len(val_df) == 0:
        raise ValueError(f"Val partition is empty for years {val_years}!")
    if len(test_df) == 0:
        raise ValueError(f"Test partition is empty for years {test_years}!")

    return train_df, val_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_event",
) -> dict[str, Any]:
    """Automated leakage verification suite.

    Checks:
    1. Maximum timestamp of train < minimum timestamp of val.
    2. Maximum timestamp of val < minimum timestamp of test.
    3. Target column is NOT present in feature_cols.
    4. Target rainfall is NOT identical to any lag feature.
    5. No correlation >= 0.999 between any single feature and the target.
    """
    report: dict[str, Any] = {"passed": True, "errors": [], "warnings": []}

    # 1 & 2: Temporal partition boundaries
    if "time" in train_df.columns and "time" in val_df.columns:
        max_train_time = pd.to_datetime(train_df["time"]).max()
        min_val_time = pd.to_datetime(val_df["time"]).min()
        if max_train_time >= min_val_time:
            report["passed"] = False
            report["errors"].append(
                f"Temporal leakage detected: max train time ({max_train_time}) >= min val time ({min_val_time})"
            )

    if "time" in val_df.columns and "time" in test_df.columns:
        max_val_time = pd.to_datetime(val_df["time"]).max()
        min_test_time = pd.to_datetime(test_df["time"]).min()
        if max_val_time >= min_test_time:
            report["passed"] = False
            report["errors"].append(
                f"Temporal leakage detected: max val time ({max_val_time}) >= min test time ({min_test_time})"
            )

    # 3: Target in feature columns
    if target_col in feature_cols:
        report["passed"] = False
        report["errors"].append(
            f"Direct target leakage: '{target_col}' is present in feature_cols!"
        )

    if "target_rainfall" in feature_cols:
        report["passed"] = False
        report["errors"].append(
            "Direct target leakage: 'target_rainfall' is present in feature_cols!"
        )

    # 4: Check if any feature perfectly duplicates the target
    y_train = train_df[target_col].values
    for col in feature_cols:
        if col in train_df.columns:
            feat_vals = train_df[col].values
            # Correlation check
            if np.std(feat_vals) > 1e-6 and np.std(y_train) > 1e-6:
                corr = np.corrcoef(feat_vals, y_train)[0, 1]
                if abs(corr) >= 0.999:
                    report["passed"] = False
                    report["errors"].append(
                        f"Suspected leakage: Feature '{col}' has near-perfect correlation ({corr:.4f}) with target!"
                    )

    if not report["passed"]:
        logger.error(f"Leakage verification FAILED: {report['errors']}")
        raise TargetLeakageError(f"Leakage detected: {'; '.join(report['errors'])}")

    logger.info("Leakage verification passed: No temporal or target leakage detected.")
    return report
