from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SequenceFitResult:
    y_pred_val: np.ndarray
    y_pred_test: np.ndarray | None
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    timing: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

