from __future__ import annotations

import numpy as np


def _as_2d(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {arr.shape}")
    return arr


def corr2_by_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Squared correlation per output delay.

    Degenerate columns with near-zero variance in either target or prediction
    receive 0.0 instead of raising, which keeps long sweeps robust.
    """
    true = _as_2d(y_true)
    pred = _as_2d(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}")

    out = np.zeros(true.shape[1], dtype=float)
    for j in range(true.shape[1]):
        target = true[:, j]
        estimate = pred[:, j]
        var_t = float(np.var(target))
        var_p = float(np.var(estimate))
        if var_t <= eps or var_p <= eps:
            out[j] = 0.0
            continue
        cov = float(np.mean((target - target.mean()) * (estimate - estimate.mean())))
        out[j] = max(0.0, cov * cov / (var_t * var_p))
    return out


def nmse_by_delay(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalized MSE per output delay.

    If a target column has near-zero variance, a perfect prediction receives
    0.0 and a non-perfect prediction receives inf. Downstream JSON helpers
    already convert non-finite values to null when persisted.
    """
    true = _as_2d(y_true)
    pred = _as_2d(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}")

    mse = np.mean((true - pred) ** 2, axis=0)
    var = np.var(true, axis=0)
    out = np.empty(true.shape[1], dtype=float)
    for j, (mse_j, var_j) in enumerate(zip(mse, var, strict=True)):
        if float(var_j) <= eps:
            out[j] = 0.0 if float(mse_j) <= eps else float("inf")
        else:
            out[j] = float(mse_j / var_j)
    return out


def memory_corr_total(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sum of squared correlations across delays."""
    return float(np.sum(corr2_by_delay(y_true, y_pred)))


def memory_eff_total(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sum of max(0, 1 - NMSE_k) across delays."""
    by_delay = nmse_by_delay(y_true, y_pred)
    return float(np.sum(np.maximum(0.0, 1.0 - by_delay)))


def max_delay_corr_above_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> int:
    """
    Largest 1-based delay whose squared correlation is at least threshold.

    Returns 0 when no delay reaches the threshold.
    """
    corr2 = corr2_by_delay(y_true, y_pred)
    hits = np.flatnonzero(corr2 >= threshold)
    if hits.size == 0:
        return 0
    return int(hits[-1] + 1)

