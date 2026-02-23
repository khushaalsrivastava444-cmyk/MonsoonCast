"""Multi-threshold and sub-regional sensitivity analysis for MonsoonCast.

Addresses Research Questions:
1. 'How do model performance and uncertainty calibration change across regions, seasons, and rainfall thresholds?'
2. 'How sensitive are the conclusions to the extreme-event definition, data resolution, and evaluation period?'

Evaluations:
1. Threshold sensitivity: 35.5, 50.0, 64.5, 100.0 mm/day (IMD meteorological classifications)
2. Sub-regional sensitivity across Central India:
   - Western Sector (73°-76°E): Arabian Sea moisture flux & leeward ghats
   - Central Core (76°-79°E): Vidarbha / Madhya Pradesh core monsoon corridor
   - Eastern Sector (79°-82°E): Bay of Bengal depression track
3. Outputs publication figure: reports/figures/fig7_threshold_subregion_sensitivity.png

Usage:
    python scripts/sensitivity_analysis.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monsooncast.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from monsooncast.evaluation.reliability import compute_expected_calibration_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sensitivity_analysis")

# Publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis across thresholds and regions.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/models", help="Model artifact directory")
    parser.add_argument("--test-data", type=str, default="data/processed/test.parquet", help="Test partition path")
    parser.add_argument("--val-data", type=str, default="data/processed/val.parquet", help="Val partition path")
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics", help="Metrics output directory")
    args = parser.parse_args()

    art_file = Path(args.artifacts_dir) / "monsooncast_models.joblib"
    if not art_file.exists():
        raise FileNotFoundError(f"Model artifacts not found at {art_file}")

    test_path = Path(args.test_data)
    val_path = Path(args.val_data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading trained models from {art_file}...")
    payload = joblib.load(art_file)
    models = payload["models"]
    model = models.get("ensemble", models.get("xgboost_classifier"))

    logger.info(f"Loading test dataset from {test_path}...")
    df_test = pd.read_parquet(test_path)
    df_val = pd.read_parquet(val_path) if val_path.exists() else None

    # Predict test probabilities once
    probs_test = model.predict_proba(df_test)
    probs_val = model.predict_proba(df_val) if df_val is not None else None

    # -------------------------------------------------------------
    # 1. THRESHOLD SENSITIVITY ANALYSIS
    # -------------------------------------------------------------
    thresholds = [35.5, 50.0, 64.5, 100.0]
    thresh_labels = ["35.5 (Rather Heavy)", "50.0 (Heavy)", "64.5 (Very Heavy)", "100.0 (Extremely Heavy)"]
    threshold_results = {}

    logger.info("Evaluating sensitivity across rainfall thresholds...")
    for tau, label in zip(thresholds, thresh_labels, strict=True):
        y_test_tau = (df_test["target_rainfall"].values >= tau).astype(int)
        event_rate = float(np.mean(y_test_tau))

        # Tune operating threshold on val partition for this rainfall threshold
        if df_val is not None:
            y_val_tau = (df_val["target_rainfall"].values >= tau).astype(int)
            opt_thresh, _ = find_optimal_threshold(y_val_tau, probs_val, metric="csi")
        else:
            opt_thresh = 0.5

        # Classification & Probabilistic metrics
        cm = compute_classification_metrics(y_test_tau, probs_test, threshold=opt_thresh)
        pr_auc = float(average_precision_score(y_test_tau, probs_test)) if event_rate > 0 else 0.0
        roc_auc = float(roc_auc_score(y_test_tau, probs_test)) if len(np.unique(y_test_tau)) > 1 else 0.5
        brier = float(brier_score_loss(y_test_tau, probs_test))
        ece = float(compute_expected_calibration_error(y_test_tau, probs_test, n_bins=10))

        threshold_results[str(tau)] = {
            "label": label,
            "threshold_mm": tau,
            "event_prevalence_pct": round(event_rate * 100, 2),
            "eval_operating_threshold": opt_thresh,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "pod": round(cm["pod"], 3),
            "far": round(cm["far"], 3),
            "csi": round(cm["csi"], 3),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
        }

    # -------------------------------------------------------------
    # 2. SUB-REGIONAL SENSITIVITY ANALYSIS (Central India Domain)
    # -------------------------------------------------------------
    logger.info("Evaluating sub-regional sensitivity across longitudinal climate sectors...")
    subregions = {
        "Western Sector (73-76E)": df_test["lon"] < 76.0,
        "Central Core (76-79E)": (df_test["lon"] >= 76.0) & (df_test["lon"] < 79.0),
        "Eastern Sector (79-82E)": df_test["lon"] >= 79.0,
    }

    subregion_results = {}
    y_test_50 = (df_test["target_rainfall"].values >= 50.0).astype(int)

    for reg_name, reg_mask in subregions.items():
        y_reg = y_test_50[reg_mask]
        p_reg = probs_test[reg_mask]
        reg_rate = float(np.mean(y_reg))

        cm_reg = compute_classification_metrics(y_reg, p_reg, threshold=0.058)
        pr_reg = float(average_precision_score(y_reg, p_reg)) if reg_rate > 0 else 0.0
        roc_reg = float(roc_auc_score(y_reg, p_reg)) if len(np.unique(y_reg)) > 1 else 0.5
        ece_reg = float(compute_expected_calibration_error(y_reg, p_reg, n_bins=10))

        subregion_results[reg_name] = {
            "n_samples": int(np.sum(reg_mask)),
            "event_prevalence_pct": round(reg_rate * 100, 2),
            "csi": round(cm_reg["csi"], 3),
            "pod": round(cm_reg["pod"], 3),
            "far": round(cm_reg["far"], 3),
            "pr_auc": round(pr_reg, 4),
            "roc_auc": round(roc_reg, 4),
            "ece": round(ece_reg, 4),
        }

    # Save to JSON
    sensitivity_payload = {
        "threshold_sensitivity": threshold_results,
        "subregion_sensitivity": subregion_results,
    }
    json_path = out_dir / "sensitivity_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sensitivity_payload, f, indent=2)
    logger.info(f"Saved sensitivity analysis results to: {json_path}")

    # -------------------------------------------------------------
    # 3. GENERATE PUBLICATION FIGURE 7
    # -------------------------------------------------------------
    logger.info("Generating Figure 7: Threshold and Sub-Regional Sensitivity...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Threshold Sensitivity
    t_vals = [threshold_results[str(tau)]["threshold_mm"] for tau in thresholds]
    csi_vals = [threshold_results[str(tau)]["csi"] for tau in thresholds]
    roc_vals = [threshold_results[str(tau)]["roc_auc"] for tau in thresholds]
    prev_vals = [threshold_results[str(tau)]["event_prevalence_pct"] for tau in thresholds]

    ax1.plot(t_vals, roc_vals, marker="s", lw=2, color="#2b6cb0", label="ROC-AUC (Discrimination)")
    ax1.plot(t_vals, csi_vals, marker="o", lw=2, color="#c53030", label="Critical Success Index (CSI)")
    ax1.set_xlabel("Rainfall Threshold (mm/day)")
    ax1.set_ylabel("Metric Score")
    ax1.set_ylim([0, 1.0])
    ax1.set_title("Forecast Skill vs Extreme-Event Threshold")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Secondary axis for prevalence
    ax1_twin = ax1.twinx()
    ax1_twin.bar(t_vals, prev_vals, width=4, alpha=0.2, color="gray", label="Base Rate (%)")
    ax1_twin.set_ylabel("Event Base Rate (%)", color="gray")
    ax1_twin.tick_params(axis="y", labelcolor="gray")
    ax1_twin.set_ylim([0, 10])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    # Panel B: Sub-Regional Performance Comparison
    reg_names_short = ["Western\n(73-76E)", "Central Core\n(76-79E)", "Eastern\n(79-82E)"]
    reg_keys = list(subregions.keys())
    csis = [subregion_results[k]["csi"] for k in reg_keys]
    rocs = [subregion_results[k]["roc_auc"] for k in reg_keys]
    eces = [subregion_results[k]["ece"] for k in reg_keys]

    x = np.arange(len(reg_names_short))
    w = 0.25

    ax2.bar(x - w, rocs, width=w, label="ROC-AUC", color="#2b6cb0", alpha=0.9)
    ax2.bar(x, csis, width=w, label="CSI", color="#c53030", alpha=0.9)
    ax2.bar(x + w, eces, width=w, label="ECE", color="#805ad5", alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(reg_names_short)
    ax2.set_ylim([0, 0.9])
    ax2.set_ylabel("Metric Score")
    ax2.set_title("Sub-Regional Verification across Climate Sectors (50 mm/day)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_fig = FIG_DIR / "fig7_threshold_subregion_sensitivity.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 7 to: {out_fig}")

    # Print summary tables
    print("\n" + "=" * 80)
    print("             MONSOONCAST THRESHOLD SENSITIVITY EVALUATION              ")
    print("=" * 80)
    print(f"{'Threshold':<24} | {'Prevalence':<10} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'CSI':<6} | {'ECE':<6}")
    print("-" * 80)
    for tau in thresholds:
        r = threshold_results[str(tau)]
        print(f"{r['label']:<24} | {r['event_prevalence_pct']:<5.2f}%    | {r['pr_auc']:<8.4f} | {r['roc_auc']:<8.4f} | {r['csi']:<6.3f} | {r['ece']:<6.3f}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("             MONSOONCAST SUB-REGIONAL SENSITIVITY EVALUATION           ")
    print("=" * 80)
    print(f"{'Sub-Region':<28} | {'N Samples':<10} | {'Prevalence':<10} | {'ROC-AUC':<8} | {'CSI':<6} | {'ECE':<6}")
    print("-" * 80)
    for k in reg_keys:
        sr = subregion_results[k]
        print(f"{k:<28} | {sr['n_samples']:<10} | {sr['event_prevalence_pct']:<5.2f}%    | {sr['roc_auc']:<8.4f} | {sr['csi']:<6.3f} | {sr['ece']:<6.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
