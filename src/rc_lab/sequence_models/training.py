from __future__ import annotations

from typing import Any

import numpy as np

from rc_lab.metrics.error import nmse, rmse
from rc_lab.metrics.memory import (
    corr2_by_delay,
    max_delay_corr_above_threshold,
    memory_corr_total,
    memory_eff_total,
    nmse_by_delay,
)


METRIC_FNS = {
    "nmse": nmse,
    "rmse": rmse,
    "corr2_by_delay": corr2_by_delay,
    "nmse_by_delay": nmse_by_delay,
    "memory_corr_total": memory_corr_total,
    "memory_eff_total": memory_eff_total,
    "max_delay_corr_above_threshold": max_delay_corr_above_threshold,
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in metric_names:
        fn = METRIC_FNS.get(name)
        if fn is None:
            raise ValueError(f"Unknown metric: {name!r}")
        metrics[name] = fn(y_true, y_pred)
    return metrics


def count_linear_readout_params(n_features: int, n_outputs: int) -> int:
    return int(n_features * n_outputs)

