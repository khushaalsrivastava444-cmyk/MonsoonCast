"""Feature importance and predictor interpretability analysis for MonsoonCast.

Addresses Research Question:
'Which meteorological predictors provide the most useful information about extreme rainfall?'

Methods:
1. XGBoost Gain and Weight (Split Frequency)
2. Random Forest Mean Decrease in Impurity (Gini Importance)
3. Model-Agnostic Permutation Feature Importance on held-out test partition
4. Generates publication figure: reports/figures/fig6_feature_importance.png

Usage:
    python scripts/analyze_feature_importance.py
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
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_feature_importance")

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


def compute_permutation_importance(
    model,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_event",
    n_repeats: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute permutation importance by measuring drop in ROC-AUC on permuted features."""
    rng = np.random.default_rng(seed)
    y_true = df_eval[target_col].values
    baseline_probs = model.predict_proba(df_eval)
    baseline_auc = float(roc_auc_score(y_true, baseline_probs))

    results = {}
    for col in feature_cols:
        perm_aucs = []
        df_perm = df_eval.copy()
        orig_vals = df_perm[col].values

        for _ in range(n_repeats):
            permuted = rng.permutation(orig_vals)
            df_perm[col] = permuted
            p_probs = model.predict_proba(df_perm)
            p_auc = float(roc_auc_score(y_true, p_probs))
            perm_aucs.append(baseline_auc - p_auc)

        results[col] = {
            "mean_delta_auc": float(np.mean(perm_aucs)),
            "std_delta_auc": float(np.std(perm_aucs)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze meteorological predictor importance.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/models", help="Model artifact directory")
    parser.add_argument("--test-data", type=str, default="data/processed/test.parquet", help="Test partition path")
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics", help="Metrics output directory")
    parser.add_argument("--sample-size", type=int, default=20000, help="Subsample size for permutation test")
    args = parser.parse_args()

    art_file = Path(args.artifacts_dir) / "monsooncast_models.joblib"
    if not art_file.exists():
        raise FileNotFoundError(f"Model artifacts not found at {art_file}")

    test_path = Path(args.test_data)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading models from {art_file}...")
    payload = joblib.load(art_file)
    models = payload["models"]
    feature_cols = payload["feature_cols"]

    logger.info(f"Loading test data from {test_path}...")
    df_test = pd.read_parquet(test_path)

    # Subsample for efficient permutation importance if large
    if len(df_test) > args.sample_size:
        logger.info(f"Subsampling {args.sample_size} instances for permutation importance...")
        df_sample = df_test.sample(n=args.sample_size, random_state=42).copy()
    else:
        df_sample = df_test.copy()

    importance_payload = {
        "feature_cols": feature_cols,
        "n_samples_evaluated": len(df_sample),
    }

    # 1. XGBoost Gain and Weight Importance
    if "xgboost_classifier" in models:
        xgb_wrapper = models["xgboost_classifier"]
        booster = xgb_wrapper.model
        score_gain = booster.get_booster().get_score(importance_type="gain")
        score_weight = booster.get_booster().get_score(importance_type="weight")

        xgb_importances = {}
        for c in feature_cols:
            xgb_importances[c] = {
                "gain": float(score_gain.get(c, 0.0)),
                "weight": float(score_weight.get(c, 0.0)),
            }
        importance_payload["xgboost"] = xgb_importances

    # 2. Random Forest Gini Importance
    if "random_forest" in models:
        rf_model = models["random_forest"].model
        rf_mdi = rf_model.feature_importances_
        rf_dict = {col: float(val) for col, val in zip(feature_cols, rf_mdi, strict=False)}
        importance_payload["random_forest"] = rf_dict

    # 3. Permutation Importance using XGBoost
    eval_model = models.get("xgboost_classifier", models.get("random_forest"))
    if eval_model:
        logger.info("Computing test-partition permutation feature importance...")
        perm_imp = compute_permutation_importance(eval_model, df_sample, feature_cols)
        importance_payload["permutation_importance"] = perm_imp

    # Save to JSON
    json_path = out_dir / "feature_importance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(importance_payload, f, indent=2)
    logger.info(f"Saved feature importance results to: {json_path}")

    # Generate Publication Figure 6: Predictor Importance
    logger.info("Generating Figure 6: Predictor Importance & Interpretability...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Permutation Importance
    if "permutation_importance" in importance_payload:
        p_imp = importance_payload["permutation_importance"]
        sorted_feats = sorted(p_imp.keys(), key=lambda k: p_imp[k]["mean_delta_auc"], reverse=True)[:12]
        means = [p_imp[k]["mean_delta_auc"] for k in sorted_feats]
        stds = [p_imp[k]["std_delta_auc"] for k in sorted_feats]

        # Categorize features by physical group
        colors = []
        for feat in sorted_feats:
            if "era5" in feat:
                colors.append("#2b6cb0")  # Blue = Atmospheric Reanalysis
            elif "spatial" in feat:
                colors.append("#38a169")  # Green = Mesoscale Spatial Context
            elif "roll" in feat or "lag" in feat or "current" in feat:
                colors.append("#dd6b20")  # Orange = Rainfall History
            else:
                colors.append("#718096")  # Gray = Climatological / Calendar

        y_pos = np.arange(len(sorted_feats))
        ax1.barh(y_pos, means, xerr=stds, align="center", color=colors, alpha=0.85, capsize=3)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_feats)
        ax1.invert_yaxis()
        ax1.set_xlabel("Mean Drop in ROC-AUC (Permutation Loss)")
        ax1.set_title("Test Set Permutation Importance (Predictive Loss)")
        ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel B: XGBoost Gain Importance
    if "xgboost" in importance_payload:
        xgb_imp = importance_payload["xgboost"]
        sorted_xgb = sorted(xgb_imp.keys(), key=lambda k: xgb_imp[k]["gain"], reverse=True)[:12]
        gains = [xgb_imp[k]["gain"] for k in sorted_xgb]

        colors_xgb = []
        for feat in sorted_xgb:
            if "era5" in feat:
                colors_xgb.append("#2b6cb0")
            elif "spatial" in feat:
                colors_xgb.append("#38a169")
            elif "roll" in feat or "lag" in feat or "current" in feat:
                colors_xgb.append("#dd6b20")
            else:
                colors_xgb.append("#718096")

        y_pos2 = np.arange(len(sorted_xgb))
        ax2.barh(y_pos2, gains, align="center", color=colors_xgb, alpha=0.85)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(sorted_xgb)
        ax2.invert_yaxis()
        ax2.set_xlabel("Relative Information Gain")
        ax2.set_title("XGBoost Relative Feature Gain")
        ax2.grid(True, linestyle="--", alpha=0.4)

    # Custom Legend for physical groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b6cb0", label="ERA5 Atmospheric State"),
        Patch(facecolor="#38a169", label="Mesoscale Spatial Context (3x3)"),
        Patch(facecolor="#dd6b20", label="Antecedent Rainfall History"),
        Patch(facecolor="#718096", label="Climatological / Calendar"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_fig = FIG_DIR / "fig6_feature_importance.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Figure 6 to: {out_fig}")

    # Print summary table
    print("\n" + "=" * 70)
    print("        MONSOONCAST METEOROLOGICAL PREDICTOR IMPORTANCE        ")
    print("=" * 70)
    print(f"{'Rank':<5} | {'Feature':<24} | {'Permutation d_AUC':<18} | {'XGB Gain':<10}")
    print("-" * 70)
    if "permutation_importance" in importance_payload and "xgboost" in importance_payload:
        ranked = sorted(
            importance_payload["permutation_importance"].keys(),
            key=lambda k: importance_payload["permutation_importance"][k]["mean_delta_auc"],
            reverse=True,
        )
        for i, feat in enumerate(ranked[:15], 1):
            d_auc = importance_payload["permutation_importance"][feat]["mean_delta_auc"]
            gain = importance_payload["xgboost"][feat]["gain"]
            print(f"{i:<5} | {feat:<24} | {d_auc:+.5f}           | {gain:<10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
