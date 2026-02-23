"""Meteorological verification scores, probabilistic calibration metrics, and bootstrap confidence intervals."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


def compute_meteorological_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard contingency table verification metrics used in operational meteorology.

    - Hits (TP)
    - False Alarms (FP)
    - Misses (FN)
    - Correct Negatives (TN)
    - POD: Probability of Detection (Recall) = TP / (TP + FN)
    - FAR: False Alarm Ratio = FP / (TP + FP)
    - CSI: Critical Success Index (Threat Score) = TP / (TP + FP + FN)
    - ETS: Equitable Threat Score
    - HSS: Heidke Skill Score
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    total = tp + fp + fn + tn

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Equitable Threat Score (ETS)
    # Expected hits by chance:
    ar = ((tp + fp) * (tp + fn)) / total if total > 0 else 0.0
    denom_ets = tp + fp + fn - ar
    ets = (tp - ar) / denom_ets if denom_ets > 0 else 0.0

    # Heidke Skill Score (HSS)
    expected_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / total if total > 0 else 0.0
    denom_hss = total - expected_correct
    hss = ((tp + tn) - expected_correct) / denom_hss if denom_hss > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pod": round(float(pod), 4),
        "far": round(float(far), 4),
        "csi": round(float(csi), 4),
        "ets": round(float(ets), 4),
        "hss": round(float(hss), 4),
    }


def compute_probabilistic_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, reference_brier: float | None = None
) -> dict[str, float]:
    """Compute probabilistic forecast verification scores.

    - Brier Score (BS): Mean squared error of probabilities
    - Brier Skill Score (BSS): 1 - BS / BS_ref (skill relative to reference climatology)
    - Log Loss: Negative log-likelihood (cross-entropy)
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)

    bs = float(brier_score_loss(y_t, y_p))

    # Brier Skill Score relative to reference climatology
    if reference_brier is not None and reference_brier > 1e-7:
        bss = 1.0 - (bs / reference_brier)
    else:
        # Default reference is sample climatology
        p_clim = float(np.mean(y_t))
        bs_clim = float(np.mean((p_clim - y_t) ** 2))
        bss = 1.0 - (bs / bs_clim) if bs_clim > 1e-7 else 0.0

    ll = float(log_loss(y_t, y_p))

    return {
        "brier_score": round(bs, 4),
        "brier_skill_score": round(float(bss), 4),
        "log_loss": round(ll, 4),
    }


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute precision, recall, F1, PR-AUC, ROC-AUC and meteorological verification."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_p >= threshold).astype(int)

    # Precision, Recall, F1
    prec = precision_score(y_t, y_pred, zero_division=0)
    rec = recall_score(y_t, y_pred, zero_division=0)
    f1 = f1_score(y_t, y_pred, zero_division=0)

    # PR-AUC
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_t, y_p)
        pr_auc = auc(r_curve, p_curve)
    except Exception:
        pr_auc = 0.0

    # ROC-AUC
    try:
        if len(np.unique(y_t)) > 1:
            roc_auc = roc_auc_score(y_t, y_p)
        else:
            roc_auc = 0.5
    except Exception:
        roc_auc = 0.5

    # Meteorological scores
    met_scores = compute_meteorological_metrics(y_t, y_pred)

    # Probabilistic scores
    prob_scores = compute_probabilistic_metrics(y_t, y_p)

    return {
        "threshold": round(threshold, 3),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        **met_scores,
        **prob_scores,
    }


def compute_rainfall_amount_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute rainfall amount metrics: MAE, RMSE, Mean Bias, and Prediction Interval coverage."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    bias = float(np.mean(y_p - y_t))

    res = {"mae": round(mae, 3), "rmse": round(rmse, 3), "bias": round(bias, 3)}

    if y_lower is not None and y_upper is not None:
        y_l = np.asarray(y_lower, dtype=float)
        y_u = np.asarray(y_upper, dtype=float)

        # Coverage: fraction of true values falling between lower and upper
        covered = (y_t >= y_l) & (y_t <= y_u)
        picp = float(np.mean(covered))
        # Width: mean interval width
        mpiw = float(np.mean(y_u - y_l))

        res["picp"] = round(picp, 4)
        res["mpiw"] = round(mpiw, 3)

    return res


def evaluate_all_metrics(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    y_true_amount: np.ndarray | None = None,
    y_pred_amount: np.ndarray | None = None,
    reference_brier: float | None = None,
) -> dict[str, Any]:
    """Unified evaluation across classification, meteorology, probability, and rainfall amount."""
    results = compute_classification_metrics(y_true_binary, y_prob, threshold=threshold)

    if reference_brier is not None:
        prob_met = compute_probabilistic_metrics(
            y_true_binary, y_prob, reference_brier=reference_brier
        )
        results["brier_skill_score"] = prob_met["brier_skill_score"]

    if y_true_amount is not None and y_pred_amount is not None:
        reg_metrics = compute_rainfall_amount_metrics(y_true_amount, y_pred_amount)
        results.update(reg_metrics)

    return results


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval (lower, point, upper) for any metric."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    point_est = float(metric_func(y_true, y_prob))

    boot_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        try:
            val = metric_func(y_true[idx], y_prob[idx])
            if not np.isnan(val):
                boot_samples.append(val)
        except Exception:
            continue

    if not boot_samples:
        return point_est, point_est, point_est

    low = float(np.percentile(boot_samples, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_samples, 100 * (1.0 - alpha / 2.0)))
    return round(low, 4), round(point_est, 4), round(high, 4)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "csi",
    n_thresholds: int = 50,
) -> tuple[float, float]:
    """Find decision threshold maximizing a meteorological metric on validation data.

    Returns:
    - (best_threshold, best_metric_value)
    Strict protocol: fit threshold on validation, never on test data.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.80, n_thresholds)
    best_thresh = 0.5
    best_score = -1.0

    for tau in thresholds:
        y_pred = (y_p >= tau).astype(int)
        if metric == "csi":
            tp = int(np.sum((y_t == 1) & (y_pred == 1)))
            fp = int(np.sum((y_t == 0) & (y_pred == 1)))
            fn = int(np.sum((y_t == 1) & (y_pred == 0)))
            score = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        elif metric == "f1":
            score = float(f1_score(y_t, y_pred, zero_division=0))
        elif metric == "ets":
            m = compute_meteorological_metrics(y_t, y_pred)
            score = m["ets"]
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_thresh = tau

    return round(float(best_thresh), 3), round(float(best_score), 4)
