"""Target formulation and extreme event labeling modules."""

from monsooncast.targets.rainfall_events import (
    build_aligned_target_series,
    compute_empirical_percentile_thresholds,
    create_binary_event_labels,
)

__all__ = [
    "build_aligned_target_series",
    "compute_empirical_percentile_thresholds",
    "create_binary_event_labels",
]
