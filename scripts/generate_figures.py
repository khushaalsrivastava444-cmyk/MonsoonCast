"""Script to generate publication-quality scientific figures for MonsoonCast research report.

Saves figures to reports/figures/:
- fig1_study_domain_climatology.png
- fig2_model_pr_roc_curves.png
- fig3_reliability_diagrams.png
- fig4_prediction_intervals.png
- fig5_spatial_forecast_uncertainty.png

Usage:
    python scripts/generate_figures.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_figures")

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


def plot_fig1_climatology(df_test: pd.DataFrame):
    """Figure 1: Study Domain and Monsoon Precipitation Climatology."""
    logger.info("Generating Figure 1: Study Domain Climatology...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

    # Map of mean rainfall
    piv = df_test.groupby(["lat", "lon"])["rain_current"].mean().unstack()
    lons = piv.columns.values
    lats = piv.index.values

    cf = ax1.contourf(lons, lats, piv.values, levels=14, cmap="Blues", extend="both")
    cbar = fig.colorbar(cf, ax=ax1, orientation="vertical", pad=0.03, shrink=0.85)
    cbar.set_label("Mean Daily Rainfall (mm/day)")
    ax1.set_title("Central India Core Monsoon Domain (18°-25°N, 73°-82°E)")
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Seasonal Progression across JJAS
    df_doy = df_test.groupby("doy")["rain_current"].agg(["mean", "max", lambda x: np.percentile(x, 95)]).reset_index()
    df_doy.columns = ["doy", "mean", "max", "p95"]

    ax2.plot(df_doy["doy"], df_doy["mean"], color="#1f77b4", lw=2, label="Domain Mean")
    ax2.plot(df_doy["doy"], df_doy["p95"], color="#ff7f0e", lw=1.8, linestyle="--", label="95th Percentile")
    ax2.plot(df_doy["doy"], df_doy["max"], color="#d62728", lw=1.5, linestyle=":", label="Daily Max Peak")
    ax2.axhline(50.0, color="gray", linestyle="-.", lw=1, label="Extreme Threshold (50 mm)")
    ax2.set_title("Intra-Seasonal Monsoon Progression (DOY 153-273)")
    ax2.set_xlabel("Day of Year (DOY)")
    ax2.set_ylabel("Rainfall (mm/day)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig1_study_domain_climatology.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_fig2_pr_roc(models: dict, df_test: pd.DataFrame):
    """Figure 2: Precision-Recall and ROC Curves."""
    logger.info("Generating Figure 2: PR and ROC Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    y_test = df_test["target_event"].values

    colors = {
        "climatology": "#7f7f7f",
        "persistence": "#8c564b",
        "logistic_regression": "#e377c2",
        "random_forest": "#1f77b4",
        "xgboost_classifier": "#2ca02c",
        "ensemble": "#9467bd"
    }

    for name in ["climatology", "persistence", "logistic_regression", "random_forest", "xgboost_classifier", "ensemble"]:
        if name in models:
            probs = models[name].predict_proba(df_test)
            p_curve, r_curve, _ = precision_recall_curve(y_test, probs)
            pr_auc = auc(r_curve, p_curve)
            ax1.plot(r_curve, p_curve, color=colors.get(name, "black"), lw=2, label=f"{name} ({pr_auc:.3f})")

            fpr, tpr, _ = roc_curve(y_test, probs)
            r_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color=colors.get(name, "black"), lw=2, label=f"{name} ({r_auc:.3f})")

    ax1.set_title("Precision-Recall Curves (Extreme Events >= 50 mm)")
    ax1.set_xlabel("Recall (Probability of Detection)")
    ax1.set_ylabel("Precision")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, label="No Skill")
    ax2.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig2_model_pr_roc_curves.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_fig3_reliability(reliability_data: dict):
    """Figure 3: Reliability Diagrams with Expected Calibration Error."""
    logger.info("Generating Figure 3: Reliability Diagrams...")
    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")

    colors = {
        "climatology": "#7f7f7f",
        "persistence": "#8c564b",
        "logistic_regression": "#e377c2",
        "random_forest": "#1f77b4",
        "xgboost_classifier": "#2ca02c",
        "ensemble": "#9467bd"
    }

    for name, d in reliability_data.items():
        if name in colors:
            ece = d.get("ece", 0.0)
            ax.plot(
                d["bin_means"], d["bin_fracs"],
                marker="o", markersize=6, lw=2,
                color=colors.get(name, "black"),
                label=f"{name} (ECE={ece:.3f})"
            )

    ax.set_title("Reliability Diagrams on Held-Out Test Seasons (2021-2023)")
    ax.set_xlabel("Mean Forecast Probability")
    ax.set_ylabel("Observed Empirical Frequency")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig3_reliability_diagrams.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_fig4_quantiles(models: dict, df_test: pd.DataFrame):
    """Figure 4: Conditional Prediction Intervals for Rainfall Amounts."""
    logger.info("Generating Figure 4: Prediction Intervals...")
    if "quantile_regressor" not in models:
        logger.warning("Quantile regressor not found. Skipping Figure 4.")
        return

    q_reg = models["quantile_regressor"]
    # Pick a representative grid cell (e.g. 21.0N, 78.0E - Nagpur Central India)
    cell_df = df_test[(df_test["lat"] == 21.0) & (df_test["lon"] == 78.0)].copy()
    if len(cell_df) == 0:
        cell_df = df_test.iloc[:122].copy()

    cell_df = cell_df.sort_values("time").iloc[:90]  # First 90 days of season
    q_preds = q_reg.predict_quantiles(cell_df)

    fig, ax = plt.subplots(figsize=(13, 5))
    times = pd.to_datetime(cell_df["time"].values)
    y_true = cell_df["target_rainfall"].values

    ax.fill_between(
        times, q_preds[0.10], q_preds[0.90],
        color="#90cdf4", alpha=0.5, label="80% Prediction Interval (Q10 - Q90)"
    )
    ax.plot(times, q_preds[0.50], color="#2b6cb0", lw=2, label="Median Forecast (Q50)")
    ax.scatter(times, y_true, color="#c53030", s=25, zorder=5, label="Observed Rainfall")
    ax.axhline(50.0, color="gray", linestyle=":", lw=1.2, label="Extreme Event (50 mm)")

    ax.set_title("Conditional Rainfall Prediction Intervals (Nagpur Region, Central India)")
    ax.set_xlabel("Forecast Valid Date")
    ax.set_ylabel("Daily Precipitation (mm/day)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig4_prediction_intervals.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_fig5_spatial_uncertainty(models: dict, df_test: pd.DataFrame):
    """Figure 5: Spatial Forecast Probability and Ensemble Spread Field."""
    logger.info("Generating Figure 5: Spatial Forecast & Uncertainty...")
    if "ensemble" not in models:
        return

    ens = models["ensemble"]
    # Pick a date with active monsoon rainfall
    active_dates = df_test[df_test["target_rainfall"] > 30]["time"].dt.strftime("%Y-%m-%d").unique()
    target_date = active_dates[len(active_dates)//2] if len(active_dates) > 0 else df_test["time"].iloc[0].strftime("%Y-%m-%d")

    day_df = df_test[df_test["time"].dt.strftime("%Y-%m-%d") == target_date].copy()
    if len(day_df) == 0:
        day_df = df_test.iloc[:285].copy()

    unc = ens.predict_with_uncertainty(day_df)
    day_df["prob"] = unc["calibrated_prob"]
    day_df["spread"] = unc["spread"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    piv_p = day_df.pivot(index="lat", columns="lon", values="prob")
    cf1 = ax1.contourf(piv_p.columns, piv_p.index, piv_p.values, levels=12, cmap="YlOrRd", vmin=0, vmax=1)
    cbar1 = fig.colorbar(cf1, ax=ax1, orientation="vertical", pad=0.03, shrink=0.85)
    cbar1.set_label("Calibrated Probability P(R >= 50 mm)")
    ax1.set_title(f"Calibrated Extreme Event Probability ({target_date})")
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    piv_s = day_df.pivot(index="lat", columns="lon", values="spread")
    cf2 = ax2.contourf(piv_s.columns, piv_s.index, piv_s.values, levels=12, cmap="Purples")
    cbar2 = fig.colorbar(cf2, ax=ax2, orientation="vertical", pad=0.03, shrink=0.85)
    cbar2.set_label(r"Ensemble Member Spread ($\sigma$)")
    ax2.set_title(f"Predictive Epistemic Uncertainty Spread ({target_date})")
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = FIG_DIR / "fig5_spatial_forecast_uncertainty.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info(f"Saved: {out}")


def main():
    art_file = Path("artifacts/models/monsooncast_models.joblib")
    test_file = Path("data/processed/test.parquet")
    rel_file = Path("artifacts/metrics/reliability_curves.json")

    if not art_file.exists() or not test_file.exists():
        raise FileNotFoundError("Artifacts missing. Please run prepare_data and train scripts first.")

    payload = joblib.load(art_file)
    models = payload["models"]
    df_test = pd.read_parquet(test_file)

    with open(rel_file, "r") as f:
        rel_data = json.load(f)

    plot_fig1_climatology(df_test)
    plot_fig2_pr_roc(models, df_test)
    plot_fig3_reliability(rel_data)
    plot_fig4_quantiles(models, df_test)
    plot_fig5_spatial_uncertainty(models, df_test)
    logger.info("All 5 publication figures successfully generated in reports/figures/")


if __name__ == "__main__":
    main()
