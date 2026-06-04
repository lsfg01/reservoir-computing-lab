"""
primitives.py — Low-level plot primitives.

All functions operate on Axes objects and are agnostic to column names.
They do NOT call apply_style() or save_figure(); callers are responsible.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_edges(vals: list[float]) -> np.ndarray:
    """Build N+1 cell-boundary edges from N centre values (handles non-uniform spacing)."""
    v = np.array(sorted(set(float(x) for x in vals)), dtype=float)
    if len(v) == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    half = np.diff(v) / 2.0
    return np.concatenate([[v[0] - half[0]], v[:-1] + half, [v[-1] + half[-1]]])


def _categorical_edges(n: int) -> np.ndarray:
    """Return cell-boundary edges for n equal-width categorical cells."""
    return np.arange(n + 1, dtype=float) - 0.5


def _fmt_tick(value: float) -> str:
    """Compact tick label that preserves 0.05/1.05 style grid values."""
    return f"{float(value):.3g}"


def _thinned_labels(vals: list[float], max_visible: int = 9) -> list[str]:
    step = max(1, (len(vals) + max_visible - 1) // max_visible)
    if step == 1:
        return [_fmt_tick(v) for v in vals]

    def _clean_score(value: float) -> int:
        # Prefer sampled values on decimal grid points over intermediate .05 cells.
        return int(np.isclose(value * 10.0, round(value * 10.0), atol=1e-8))

    def _offset_score(offset: int) -> tuple[int, int, int]:
        chosen = [float(v) for i, v in enumerate(vals) if (i - offset) % step == 0]
        clean = sum(_clean_score(v) for v in chosen)
        includes_one = int(any(np.isclose(v, 1.0, atol=1e-8) for v in chosen))
        # Keep offset 0 as the tie-breaker when cleanliness is equal.
        return clean, includes_one, -offset

    offset = max(range(step), key=_offset_score)
    return [_fmt_tick(v) if (i - offset) % step == 0 else "" for i, v in enumerate(vals)]


def categorical_index(vals: list[float], value: float) -> float:
    """
    Map a real sampled-axis value to its categorical cell-centre coordinate.

    Values between samples are linearly interpolated in index space.
    """
    ordered = np.array(sorted(set(float(x) for x in vals)), dtype=float)
    if len(ordered) == 0:
        raise ValueError("Cannot map values on an empty categorical axis")
    if len(ordered) == 1:
        return 0.0
    positions = np.arange(len(ordered), dtype=float)
    return float(np.interp(float(value), ordered, positions))


def categorical_vline(
    ax: "plt.Axes",
    x_ticks: list[float],
    value: float,
    **kwargs: Any,
) -> None:
    """Draw a vertical reference line at a real value on a categorical x-axis."""
    ax.axvline(categorical_index(x_ticks, value), **kwargs)


# ---------------------------------------------------------------------------
# Public primitives
# ---------------------------------------------------------------------------

def heatmap_plane(
    ax: "plt.Axes",
    grid2d: np.ndarray,
    x_ticks: list[float],
    y_ticks: list[float],
    cmap: str,
    vmin: float,
    vmax: float,
    mask: np.ndarray | None = None,
    annotate: bool = False,
) -> Any:
    """
    Draw a categorical heatmap via pcolormesh with ticks at cell centres.

    grid2d shape: (len(y_ticks), len(x_ticks)).
    y_ticks must be in descending order (as returned by pivot_plane);
    they are flipped internally so y increases upward on the axes.

    Each sampled x/y value receives one equal-width cell. Real grid values are
    shown as tick labels; overlays should use categorical_index().

    Fine grid lines (lightgray, lw=0.3) are drawn at cell boundaries.
    x-axis tick labels are thinned to at most 9 visible labels and rotated
    45° to prevent crowding.

    mask (bool array, same shape as grid2d): True cells are attenuated
    with a semi-transparent white overlay.

    Returns the pcolormesh ScalarMappable for colorbar attachment.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    y_asc = list(reversed(y_ticks))
    grid_asc = grid2d[::-1].copy()

    x_centers = np.arange(len(x_ticks), dtype=float)
    y_centers = np.arange(len(y_asc), dtype=float)
    x_edges = _categorical_edges(len(x_ticks))
    y_edges = _categorical_edges(len(y_asc))

    pcm = ax.pcolormesh(
        x_edges, y_edges, grid_asc,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
    )

    # Clip view exactly to data extent (prevents matplotlib from padding)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    # Fine grid at cell boundaries (draw before patches so it sits behind)
    for xe in x_edges:
        ax.axvline(xe, color="lightgray", lw=0.3, zorder=2)
    for ye in y_edges:
        ax.axhline(ye, color="lightgray", lw=0.3, zorder=2)

    # Mask overlay (semi-transparent white)
    if mask is not None:
        mask_asc = mask[::-1]
        patches = []
        for i in range(len(y_asc)):
            for j in range(len(x_ticks)):
                if mask_asc[i, j]:
                    patches.append(Rectangle(
                        (x_edges[j], y_edges[i]),
                        1.0,
                        1.0,
                    ))
        if patches:
            pc = PatchCollection(
                patches, facecolor="white", alpha=0.55,
                edgecolor="none", zorder=3,
            )
            ax.add_collection(pc)

    if annotate:
        for i, y_val in enumerate(y_asc):
            for j, x_val in enumerate(x_ticks):
                val = float(grid_asc[i, j])
                if np.isfinite(val):
                    ax.text(x_centers[j], y_centers[i], f"{val:.2g}",
                            ha="center", va="center", fontsize=6, zorder=5)

    # x ticks: place at cell centres; thin to ≤9 visible labels, rotate 45°
    ax.set_xticks(x_centers)
    ax.set_xticklabels(_thinned_labels(x_ticks), rotation=45, ha="right", fontsize=7)

    # y ticks: all labels, cell centres
    ax.set_yticks(y_centers)
    ax.set_yticklabels(_thinned_labels(y_asc), fontsize=7)

    return pcm


def overlay_frontier(
    ax: "plt.Axes",
    x_ticks: list[float],
    frontier_per_x: dict[float, float],
) -> None:
    """
    Draw the ESP frontier as a step-line.

    frontier_per_x maps x -> max_y (the rightmost stable x or highest stable y
    depending on orientation). Only x values present in both x_ticks and
    frontier_per_x are drawn.
    """
    xs = sorted(x for x in x_ticks if x in frontier_per_x)
    ys = [frontier_per_x[x] for x in xs]
    if xs:
        ax.step(xs, ys, where="post", color="#d62728",
                lw=1.8, ls="--", zorder=5, label="frontera ESP")


def overlay_frontier_per_y(
    ax: "plt.Axes",
    x_ticks: list[float],
    y_ticks: list[float],
    frontier_per_y: dict[float, float],
) -> None:
    """
    Draw rho*(y) as the right edge of each categorical heatmap row.

    frontier_per_y maps each real y value to the largest synchronized real x.
    The drawn boundary sits at xidx(rho*) + 0.5, the right cell edge.
    """
    y_asc = sorted(float(v) for v in y_ticks)
    if not y_asc or not frontier_per_y:
        return

    y_keys = list(frontier_per_y.keys())
    x_edges: list[float] = []
    for y_val in y_asc:
        best_y = min(y_keys, key=lambda k: abs(float(k) - y_val))
        if abs(float(best_y) - y_val) > 1e-6:
            continue
        x_edges.append(categorical_index(x_ticks, frontier_per_y[best_y]) + 0.5)

    if not x_edges:
        return

    line_kw = dict(
        color="#d62728",
        lw=1.8,
        ls="--",
        zorder=5,
        solid_capstyle="butt",
        dash_capstyle="butt",
    )
    for idx, x_edge in enumerate(x_edges):
        ax.plot(
            [x_edge, x_edge],
            [idx - 0.5, idx + 0.5],
            label="frontera ESP" if idx == 0 else None,
            **line_kw,
        )
        if idx < len(x_edges) - 1 and not np.isclose(x_edge, x_edges[idx + 1]):
            ax.plot(
                [x_edge, x_edges[idx + 1]],
                [idx + 0.5, idx + 0.5],
                **line_kw,
            )


def overlay_regions(
    ax: "plt.Axes",
    regions: list[dict[str, Any]],
    x_ticks: list[float] | None = None,
    y_ticks: list[float] | None = None,
) -> None:
    """
    Draw named rectangular regions.

    regions: list of {name: str, x: (lo, hi), y: (lo, hi)}.
    If x_ticks/y_ticks are provided, bounds are mapped to categorical cell
    edges: idx(lo)-0.5 ... idx(hi)+0.5.
    """
    from matplotlib.patches import Rectangle

    _colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for idx, r in enumerate(regions):
        x_lo, x_hi = r["x"]
        y_lo, y_hi = r["y"]
        color = _colors[idx % len(_colors)]
        if x_ticks is not None and y_ticks is not None:
            x0 = categorical_index(x_ticks, x_lo) - 0.5
            x1 = categorical_index(x_ticks, x_hi) + 0.5
            y0 = categorical_index(y_ticks, y_lo) - 0.5
            y1 = categorical_index(y_ticks, y_hi) + 0.5
        else:
            x0, x1 = x_lo, x_hi
            y0, y1 = y_lo, y_hi
        rect = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.5, edgecolor=color, facecolor="none",
            linestyle="-", alpha=0.75, zorder=4,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + 0.02 * (x1 - x0), y1 - 0.04 * (y1 - y0),
            r["name"], fontsize=7, color=color,
            va="top", ha="left", zorder=6,
        )


def lines_by_group(
    ax: "plt.Axes",
    df: "pd.DataFrame",
    x_col: str,
    y_col: str,
    group_col: str,
    logy: bool = False,
) -> None:
    """
    Draw one line per unique value of group_col; NaN y-values are skipped.
    """
    from rc_lab.viz.style import PALETTE

    groups = sorted(df[group_col].dropna().unique())
    for idx, g in enumerate(groups):
        sub = df[df[group_col] == g].sort_values(x_col)
        valid = sub[y_col].notna()
        ax.plot(
            sub.loc[valid, x_col].to_numpy(),
            sub.loc[valid, y_col].to_numpy(),
            label=f"{group_col}={g}",
            color=PALETTE[idx % len(PALETTE)],
            marker="o", ms=3, lw=1.5,
        )
    if logy:
        ax.set_yscale("log")


def add_colorbar(
    fig: "plt.Figure",
    ax: "plt.Axes",
    mappable: Any,
    label: str,
) -> Any:
    """Attach a colorbar to ax and return it."""
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=8)
    return cb


def surface_plane(
    ax3d: Any,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    """
    Draw a 3D surface. Cells where mask is True are set to NaN (not rendered).
    X, Y, Z must be 2D arrays of identical shape (as returned by np.meshgrid).
    """
    Z_plot = Z.copy().astype(float)
    if mask is not None:
        Z_plot[mask] = np.nan
    ax3d.plot_surface(X, Y, Z_plot, cmap="viridis", alpha=0.85, edgecolor="none")
