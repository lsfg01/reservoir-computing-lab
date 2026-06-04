"""
io.py — Tabular data loading and pivot utilities.

Uses pandas as the tabular interface; works with any runner's summary.csv.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_summary(csv_path: str | Path) -> pd.DataFrame:
    """Load any runner summary CSV as a DataFrame."""
    return pd.read_csv(csv_path)


def pivot_plane(
    df: pd.DataFrame,
    value_col: str,
    x_col: str,
    y_col: str,
    fixed: dict[str, Any],
) -> tuple[np.ndarray, list[float], list[float], np.ndarray]:
    """
    Pivot df[value_col] into a 2D grid (rows = y descending, cols = x ascending).

    fixed filters the DataFrame before pivoting (e.g. {'alpha': 1.0, 'family_name': 'random_sparse'}).
    Float values in fixed are compared with np.isclose.

    Returns
    -------
    grid2d : np.ndarray, shape (len(y_sorted_desc), len(x_sorted_asc))
    x_sorted : list[float], ascending
    y_sorted : list[float], descending
    mask     : bool ndarray, True where grid2d is NaN
    """
    sub = df.copy()
    for k, v in fixed.items():
        col = sub[k]
        if isinstance(v, float):
            sub = sub[np.isclose(col.astype(float).to_numpy(), v)]
        else:
            sub = sub[col == v]

    x_sorted: list[float] = sorted(float(v) for v in sub[x_col].unique())
    y_sorted: list[float] = sorted((float(v) for v in sub[y_col].unique()), reverse=True)

    xi: dict[float, int] = {v: i for i, v in enumerate(x_sorted)}
    yi: dict[float, int] = {v: i for i, v in enumerate(y_sorted)}

    grid2d = np.full((len(y_sorted), len(x_sorted)), np.nan)
    for _, row in sub.iterrows():
        xv = float(row[x_col])
        yv = float(row[y_col])
        # Match via closest key (tolerates CSV float rounding)
        best_x = min(xi.keys(), key=lambda k: abs(k - xv))
        best_y = min(yi.keys(), key=lambda k: abs(k - yv))
        if abs(best_x - xv) < 1e-6 and abs(best_y - yv) < 1e-6:
            val = row[value_col]
            grid2d[yi[best_y], xi[best_x]] = float(val) if pd.notna(val) else np.nan

    return grid2d, x_sorted, y_sorted, np.isnan(grid2d)


def frontier_per_x(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    valid_col: str,
    thresh: float,
    fixed: dict[str, Any],
) -> dict[float, float]:
    """
    For each x value, find the maximum y where valid_col >= thresh.

    Returns dict {x_val: max_y_val}. x or y values not meeting thresh are absent.
    """
    sub = df.copy()
    for k, v in fixed.items():
        col = sub[k]
        if isinstance(v, float):
            sub = sub[np.isclose(col.astype(float).to_numpy(), v)]
        else:
            sub = sub[col == v]

    valid = sub[sub[valid_col].astype(float) >= thresh]
    result: dict[float, float] = {}
    for x_val, group in valid.groupby(x_col):
        result[float(x_val)] = float(group[y_col].max())
    return result


def frontier_per_y(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    valid_col: str,
    thresh: float,
    fixed: dict[str, Any],
) -> dict[float, float]:
    """
    For each y value, find the maximum x where valid_col >= thresh.

    Rows with no valid x are assigned the minimum sampled x. This mirrors the
    frontier convention used by the ESP figures: the right edge of the first
    sampled cell is the fallback boundary when no cell is synchronized.

    Returns dict {y_val: max_x_val}.
    """
    sub = df.copy()
    for k, v in fixed.items():
        col = sub[k]
        if isinstance(v, float):
            sub = sub[np.isclose(col.astype(float).to_numpy(), v)]
        else:
            sub = sub[col == v]

    x_values = sorted(float(v) for v in sub[x_col].unique())
    y_values = sorted(float(v) for v in sub[y_col].unique())
    if not x_values or not y_values:
        return {}

    fallback_x = min(x_values)
    result: dict[float, float] = {}
    y_numeric = sub[y_col].astype(float).to_numpy()
    for y_val in y_values:
        group = sub[np.isclose(y_numeric, y_val)]
        valid = group[group[valid_col].astype(float) >= thresh]
        if len(valid) > 0:
            result[y_val] = float(valid[x_col].astype(float).max())
        else:
            result[y_val] = fallback_x
    return result


def frontier_per_sin(
    df: pd.DataFrame,
    rho_col: str,
    sin_col: str,
    valid_col: str,
    thresh: float,
    fixed: dict[str, Any],
) -> dict[float, float]:
    """Convenience wrapper for rho*(s_in): {s_in: max synchronized rho}."""
    return frontier_per_y(df, rho_col, sin_col, valid_col, thresh, fixed)


def load_point_curves(
    runs_dir: str | Path,
    selector: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Load per-point JSON files from runs_dir that match selector.

    Float values are compared with tolerance 1e-4; strings with equality.
    Returns list of dicts (each contains 'd_curve_mean_sub' and metadata).
    """
    runs_dir = Path(runs_dir)
    results: list[dict[str, Any]] = []

    for json_path in sorted(runs_dir.glob("*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        match = True
        for k, v in selector.items():
            dv = data.get(k)
            if isinstance(v, float) and isinstance(dv, (int, float)):
                if abs(float(dv) - v) > 1e-4:
                    match = False
                    break
            elif str(dv) != str(v):
                match = False
                break

        if match:
            results.append(data)

    return results
