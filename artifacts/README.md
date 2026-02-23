# Model Artifacts and Experiment Registry

This directory contains versioned model weights, calibration scalers, feature encoders, evaluation metrics, and experiment metadata.

## Directory Structure

```
artifacts/
├── README.md
├── experiments/            # JSON records of reproducible experiment runs
├── metrics/                # Metric summaries (classification, probabilistic, regression)
├── figures/                # Reliability diagrams, PR curves, and spatial plots
└── models/                 # Serialized model pipelines (.joblib and .pt checkpoints)
```

## Traceability Requirements

Every saved model artifact in `artifacts/models/` must correspond to a JSON entry in `artifacts/experiments/` containing:
- Unique `experiment_id` (UUID + timestamp)
- Configuration SHA-256 hash
- Dataset version and spatial/temporal split bounds
- Extreme-event threshold definition ($\tau$)
- Calibration method and fit partition
- Block-bootstrapped validation & test metrics
