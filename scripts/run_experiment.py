"""Master experiment runner for MonsoonCast.

Runs end-to-end pipeline:
1. Prepare features and chronological splits
2. Train baselines, classical ML models, and calibrated ensemble
3. Evaluate on held-out test data with bootstrap confidence intervals
4. Register experiment manifest in artifacts/experiments/

Usage:
    python scripts/run_experiment.py [--config configs/baseline.yaml]
"""

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monsooncast.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_experiment")


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end MonsoonCast experiment.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_id = f"{cfg.id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path("artifacts/experiments")
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting MonsoonCast Experiment: {exp_id}")
    logger.info(f"Configuration Hash: {cfg.compute_hash()}")

    # Step 1: Prepare Data
    logger.info(">>> Step 1/3: Preparing data and chronological partitions...")
    cmd_prep = [sys.executable, "scripts/prepare_data.py", "--config", cfg.data_config_path]
    if getattr(cfg.features, "include_reanalysis_predictors", False):
        logger.info("Atmospheric reanalysis enabled in config; including ERA5 predictors.")
        cmd_prep.append("--include-era5")
    subprocess.run(cmd_prep, check=True)

    # Step 2: Train Models
    logger.info(">>> Step 2/3: Training models and calibrating ensemble...")
    cmd_train = [sys.executable, "scripts/train.py", "--config", args.config]
    subprocess.run(cmd_train, check=True)

    # Step 3: Evaluate Models
    logger.info(">>> Step 3/3: Evaluating models on held-out test set...")
    cmd_eval = [sys.executable, "scripts/evaluate.py"]
    subprocess.run(cmd_eval, check=True)

    # Record experiment registry
    results_file = Path("artifacts/metrics/evaluation_results.json")
    metrics_summary = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            metrics_summary = json.load(f)

    manifest = {
        "experiment_id": exp_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config_hash": cfg.compute_hash(),
        "config_path": args.config,
        "seed": cfg.seed,
        "train_years": cfg.data.temporal_scope.train_years if cfg.data else [],
        "val_years": cfg.data.temporal_scope.val_years if cfg.data else [],
        "test_years": cfg.data.temporal_scope.test_years if cfg.data else [],
        "threshold": cfg.data.target.default_threshold if cfg.data else 50.0,
        "metrics_summary": metrics_summary,
        "model_artifact": "artifacts/models/monsooncast_models.joblib"
    }

    manifest_path = exp_dir / f"{exp_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Experiment {exp_id} successfully registered at {manifest_path}")


if __name__ == "__main__":
    main()
