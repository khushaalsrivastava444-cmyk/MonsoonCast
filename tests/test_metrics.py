"""Unit tests for meteorological verification and probabilistic evaluation metrics."""

import numpy as np

from monsooncast.evaluation.metrics import (
    compute_meteorological_metrics,
    compute_probabilistic_metrics,
    compute_rainfall_amount_metrics,
)
from monsooncast.evaluation.reliability import (
    compute_expected_calibration_error,
)


def test_contingency_metrics_exact():
    """Verify POD, FAR, CSI against known manual toy truth.

    Truth:  [1, 1, 0, 0, 1]
    Pred:   [1, 0, 1, 0, 1]

    TP: 2 (indices 0, 4)
    FP: 1 (index 2)
    FN: 1 (index 1)
    TN: 1 (index 3)

    POD = TP / (TP + FN) = 2 / 3 = 0.6667
    FAR = FP / (TP + FP) = 1 / 3 = 0.3333
    CSI = TP / (TP + FP + FN) = 2 / 4 = 0.5000
    """
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    m = compute_meteorological_metrics(y_true, y_pred)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert np.isclose(m["pod"], 2.0 / 3.0, atol=1e-3)
    assert np.isclose(m["far"], 1.0 / 3.0, atol=1e-3)
    assert np.isclose(m["csi"], 0.50, atol=1e-3)


def test_brier_score_and_skill():
    """Verify Brier Score on perfect and imperfect predictions."""
    y_true = np.array([1, 0, 1, 0])
    y_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    y_unskilled = np.array([0.5, 0.5, 0.5, 0.5])

    res_perf = compute_probabilistic_metrics(y_true, y_perfect)
    assert res_perf["brier_score"] == 0.0

    res_unskilled = compute_probabilistic_metrics(y_true, y_unskilled)
    assert np.isclose(res_unskilled["brier_score"], 0.25, atol=1e-3)


def test_expected_calibration_error():
    """Verify ECE is 0 for perfectly calibrated predictions."""
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array(
        [0.5, 0.5, 0.5, 0.5]
    )  # In bin [0.4, 0.6], mean prob = 0.5, mean true = 0.5 -> diff = 0
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert np.isclose(ece, 0.0, atol=1e-4)


def test_prediction_interval_coverage():
    """Verify PICP coverage calculation."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_lower = np.array([5.0, 15.0, 25.0, 50.0])  # 3 of 4 covered
    y_upper = np.array([15.0, 25.0, 35.0, 60.0])
    y_pred = np.array([10.0, 20.0, 30.0, 55.0])

    res = compute_rainfall_amount_metrics(y_true, y_pred, y_lower=y_lower, y_upper=y_upper)
    assert res["picp"] == 0.75
