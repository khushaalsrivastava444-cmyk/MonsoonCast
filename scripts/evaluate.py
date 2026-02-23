"""Evaluation pipeline for comparing baselines, ML models, and calibrated ensembles.

Usage:
    python scripts/evaluate.py [--artifacts-dir artifacts/models] [--test-data data/processed/test.parquet]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monsooncast.evaluation.metrics import (
    compute_classification_metrics,
    compute_probabilistic_metrics,
    compute_rainfall_amount_metrics,
    evaluate_all_metrics,
    bootstrap_metric_ci,
)
from monsooncast.evaluation.reliability import compute_reliability_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MonsoonCast models on held-out test data.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/models", help="Directory containing model artifacts")
    parser.add_argument("--test-data", type=str, default="data/processed/test.parquet", help="Path to test partition")
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics", help="Directory to save evaluation results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for probability")
    args = parser.parse_args()

    art_file = Path(args.artifacts_dir) / "monsooncast_models.joblib"
    if not art_file.exists():
        raise FileNotFoundError(f"Trained model artifacts not found at {art_file}. Run 'python scripts/train.py' first.")

    test_path = Path(args.test_data)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading trained models from {art_file}...")
    payload = joblib.load(art_file)
    models = payload["models"]
    feature_cols = payload["feature_cols"]

    logger.info(f"Loading held-out test dataset from {test_path}...")
    df_test = pd.read_parquet(test_path)
    y_test_binary = df_test["target_event"].values
    y_test_amount = df_test["target_rainfall"].values

    logger.info(
        f"Evaluating on {len(df_test)} test examples. "
        f"Extreme event frequency: {np.mean(y_test_binary)*100:.2f}%"
    )

    # Compute reference Brier score of Climatology baseline for BSS
    clim_probs = models["climatology"].predict_proba(df_test)
    ref_brier = float(np.mean((clim_probs - y_test_binary) ** 2))

    comparison_results = {}
    reliability_results = {}

    # Evaluate each classification / ensemble model
    eval_model_names = [m for m in models.keys() if m not in ("quantile_regressor", "ridge_regressor")]

    for name in eval_model_names:
        logger.info(f"Evaluating {name}...")
        model = models[name]
        probs = model.predict_proba(df_test)

        # Standard + meteorological + probabilistic metrics
        metrics = evaluate_all_metrics(
            y_true_binary=y_test_binary,
            y_prob=probs,
            threshold=args.threshold,
            reference_brier=ref_brier
        )

        # Reliability curve & ECE
        rel_curve = compute_reliability_curve(y_test_binary, probs, n_bins=10)
        metrics["ece"] = rel_curve["ece"]
        reliability_results[name] = rel_curve

        # Bootstrap confidence interval for CSI (Threat Score)
        csi_low, csi_pt, csi_high = bootstrap_metric_ci(
            y_test_binary,
            probs,
            metric_func=lambda yt, yp: compute_classification_metrics(yt, yp, threshold=args.threshold)["csi"],
            n_bootstrap=200
        )
        metrics["csi_95ci"] = [csi_low, csi_high]

        comparison_results[name] = metrics

    # Evaluate Quantile Regressor if present
    if "quantile_regressor" in models:
        logger.info("Evaluating XGBoost Quantile Regressor prediction intervals...")
        q_reg = models["quantile_regressor"]
        q_preds = q_reg.predict_quantiles(df_test)
        if 0.10 in q_preds and 0.90 in q_preds and 0.50 in q_preds:
            q_metrics = compute_rainfall_amount_metrics(
                y_true=y_test_amount,
                y_pred=q_preds[0.50],
                y_lower=q_preds[0.10],
                y_upper=q_preds[0.90]
            )
            comparison_results["quantile_regressor"] = q_metrics

    # Save to JSON
    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    reliability_path = out_dir / "reliability_curves.json"
    with open(reliability_path, "w", encoding="utf-8") as f:
        json.dump(reliability_results, f, indent=2)

    # Print publication-quality scientific table
    print("\n" + "=" * 90)
    print("                MONSOONCAST RESEARCH EVALUATION RESULTS                ")
    print("=" * 90)
    header = f"{'Model':<24} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'POD':<6} | {'FAR':<6} | {'CSI (95% CI)':<16} | {'BSS':<7} | {'ECE':<6}"
    print(header)
    print("-" * 90)

    for m_name, m_res in comparison_results.items():
        if "pr_auc" in m_res:
            ci_str = f"{m_res.get('csi', 0):.3f} [{m_res.get('csi_95ci', [0,0])[0]:.2f}-{m_res.get('csi_95ci', [0,0])[1]:.2f}]"
            row = (
                f"{m_name:<24} | "
                f"{m_res.get('pr_auc', 0):.4f}   | "
                f"{m_res.get('roc_auc', 0):.4f}   | "
                f"{m_res.get('pod', 0):.3f}  | "
                f"{m_res.get('far', 0):.3f}  | "
                f"{ci_str:<16} | "
                f"{m_res.get('brier_skill_score', 0):+.3f}  | "
                f"{m_res.get('ece', 0):.3f}"
            )
            print(row)
    print("=" * 90)
    logger.info(f"Results successfully saved to {results_path} and {reliability_path}")


if __name__ == "__main__":
    main()
