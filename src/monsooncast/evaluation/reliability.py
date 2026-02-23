"""Probability calibration curves, reliability diagrams, and expected calibration error."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Compute reliability diagram statistics: binned confidence vs observed empirical frequency.

    Bins predictions into n_bins between [0.0, 1.0].
    For each bin:
    - mean_pred: mean predicted probability
    - frac_pos: observed positive fraction
    - count: number of samples in bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Clip probabilities to [0, 1]
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    # Handle upper boundary edge case (prob == 1.0)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means = []
    bin_fracs = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            bin_means.append(float(np.mean(y_prob[mask])))
            bin_fracs.append(float(np.mean(y_true[mask])))
        else:
            # Midpoint of empty bin
            bin_means.append(float(0.5 * (bins[i] + bins[i + 1])))
            bin_fracs.append(np.nan)

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return {
        "bins": bins.tolist(),
        "bin_means": bin_means,
        "bin_fracs": bin_fracs,
        "bin_counts": bin_counts,
        "ece": round(ece, 4),
    }


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE):

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)
    if N == 0:
        return 0.0

    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        if count > 0:
            mean_prob = np.mean(y_prob[mask])
            mean_true = np.mean(y_true[mask])
            ece += (count / N) * np.abs(mean_true - mean_prob)

    return float(ece)
