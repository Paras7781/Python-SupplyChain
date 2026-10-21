"""Shared summaries for cross-validation metric estimates."""
from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np


def summarize_cv_results(
    results: Mapping[str, object],
    metric_names: Collection[str],
    negative_metrics: Collection[str] = (),
) -> dict[str, dict[str, float | list[float]]]:
    """Return fold scores, mean, and sample standard deviation for each metric."""
    summaries: dict[str, dict[str, float | list[float]]] = {}
    for metric in metric_names:
        scores = np.asarray(results[f"test_{metric}"], dtype=float)
        if metric in negative_metrics:
            scores = -scores
        if len(scores) < 2 or not np.isfinite(scores).all():
            raise ValueError(
                f"Cross-validation metric '{metric}' must have at least two finite fold scores."
            )
        summaries[metric] = {
            "fold_scores": [round(float(score), 4) for score in scores],
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std(ddof=1)), 4),
        }
    return summaries
