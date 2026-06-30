"""
frontier.py — Figure builders for the ESP frontier study (Paso 0).

build_frontier_figures() generates F1–F8 and returns all written paths.

Label convention:
  - saturation_mean_mean field = mean state amplitude ⟨|x|⟩ (not saturation).
    It is relabelled in all figures; the CSV field name is unchanged.
  - saturation_frac_mean = fraction of states with |x|>0.99 (true saturation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rc_lab.viz.io import frontier_per_sin, frontier_per_x, load_point_curves, load_summary, pivot_plane
from rc_lab.viz.primitives import (
    add_colorbar,
    categorical_index,
    categorical_vline,
    heatmap_plane,
    overlay_frontier_per_y,
    surface_plane,  # kept in public API; not used by F7 after rewrite
)
from rc_lab.viz.style import CMAPS, PALETTE, apply_style, save_figure

# Theoretical regions are conceptual dense subsets of parameter space. The
# experimental A/B1/B2/C grids below are their finite sweep discretization.
_THEORETICAL_REGIONS: dict[str, dict[str, Any]] = {
    "R1": {
        "rho": (0.60, 0.90),
        "input_scaling": (0.05, 0.20),
        "leak_rate": (0.90, 1.00),
        "label": "R1",
    },
    "R2": {
        "rho": (0.90, 1.05),
        "input_scaling": (0.05, 0.20),
        "leak_rate": (0.90, 1.00),
        "label": "R2",
    },
    "R3": {
        "rho": (1.00, 1.50),
        "input_scaling": (0.40, 1.50),
        "leak_rate": (0.90, 1.00),
        "label": "R3",
    },
    "R4": {
        "rho": (0.80, 1.40),
        "input_scaling": (0.40, 1.50),
        "leak_rate": (0.30, 0.50),
        "label": "R4",
    },
}

# Backward-compatible 2-D representation accepted by F2/F3 and --regions.
CANDIDATE_REGIONS: list[dict[str, Any]] = [
    {
        "name": name,
        "x": spec["rho"],
        "y": spec["input_scaling"],
        "alpha": spec["leak_rate"],
    }
    for name, spec in _THEORETICAL_REGIONS.items()
]

_SYNC_THRESH = 0.999
_TRUNC_THRESH = 0.5   # nonsync_fraction_descending >= this → truncation marker

# Axis label strings (mathtext)
_LAB_RHO    = r"radio espectral $\rho$"
_LAB_SIN    = r"escala entrada $s_{\mathrm{in}}$"
_LAB_ALPHA  = r"$\alpha$ (leak rate)"
_LAB_WASHOUT = r"washout empírico (pasos)"
_LAB_AMP     = r"amplitud media $\langle|x|\rangle$"
_LAB_FRAC    = "frac. sincronización"
_LAB_LOGW    = r"$\log_{10}$(washout)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _family_name(df: Any) -> str:
    return str(df["family_name"].dropna().unique()[0])


def _fixed_alpha(family: str, alpha: float) -> dict[str, Any]:
    return {"family_name": family, "alpha": alpha}


def _isclose_filter(df: Any, col: str, val: float) -> Any:
    return df[np.isclose(df[col].astype(float).to_numpy(), val)]


def _log_grid(grid: np.ndarray) -> np.ndarray:
    """Log10 of strictly-positive cells; NaN elsewhere."""
    return np.where(grid > 0.0, np.log10(np.maximum(grid, 1e-1)), np.nan)


def _sync_mask(df: Any, x_col: str, y_col: str, fixed: dict[str, Any]) -> np.ndarray:
    """Bool mask (rows=y_desc, cols=x_asc) where frac_sync < _SYNC_THRESH."""
    sg, _, _, _ = pivot_plane(df, "fraction_synchronized_mean", x_col, y_col, fixed)
    return sg < _SYNC_THRESH


def _nfd_float(row: Any) -> float:
    try:
        return float(row.get("nonsync_fraction_descending") if hasattr(row, "get") else row["nonsync_fraction_descending"])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def build_theoretical_regions() -> dict[str, dict[str, Any]]:
    """Return independent 3-D definitions of the conceptual R1-R4 regions."""
    return {
        name: {
            "rho": tuple(spec["rho"]),
            "input_scaling": tuple(spec["input_scaling"]),
            "leak_rate": tuple(spec["leak_rate"]),
            "label": str(spec["label"]),
        }
        for name, spec in _THEORETICAL_REGIONS.items()
    }


def _grid_points(
    spectral_radius: list[float],
    input_scaling: list[float],
    leak_rate: list[float],
) -> list[dict[str, float]]:
    return [
        {
            "spectral_radius": float(rho),
            "input_scaling": float(s_in),
            "leak_rate": float(alpha),
        }
        for rho in spectral_radius
        for s_in in input_scaling
        for alpha in leak_rate
    ]


def build_experimental_grids() -> dict[str, list[dict[str, float]]]:
    """Return the 88 A/B1/B2/C config points used by the later sweep."""
    return {
        "A": _grid_points(
            [0.8, 0.9, 0.95, 1.0, 1.05],
            [0.05, 0.1, 0.2],
            [0.9, 1.0],
        ),
        "B1": _grid_points(
            [1.0, 1.2, 1.4],
            [0.4, 0.8, 1.0],
            [0.9, 1.0],
        ),
        "B2": _grid_points(
            [1.0, 1.2, 1.4, 1.5],
            [1.5],
            [0.9, 1.0],
        ),
        "C": _grid_points(
            [0.8, 1.0, 1.2, 1.4],
            [0.4, 0.8, 1.0, 1.5],
            [0.3, 0.5],
        ),
    }


def temporal_extension_rhos(
    group: Any,
    rho_star: float,
    descending_threshold: float = _TRUNC_THRESH,
) -> list[float]:
    """Return the consecutive exterior p_next values with censor evidence."""
    above = group[
        group["rho_target"].astype(float) > float(rho_star)
    ].sort_values("rho_target")
    extension: list[float] = []
    for _, row in above.iterrows():
        nfd = _nfd_float(row)
        if not np.isfinite(nfd) or nfd < descending_threshold:
            break
        extension.append(float(row["rho_target"]))
    return extension


def detect_temporal_truncation(
    group: Any,
    rho_star: float,
    descending_threshold: float = _TRUNC_THRESH,
) -> tuple[bool, float | None]:
    """Return whether a temporal tail exists and its final p_next."""
    extension = temporal_extension_rhos(group, rho_star, descending_threshold)
    return bool(extension), extension[-1] if extension else None


def detect_spatial_truncation(rho_star: float, rho_max: float) -> bool:
    """Return whether the observed frontier reaches the rho grid boundary."""
    return bool(np.isclose(float(rho_star), float(rho_max), atol=1e-8))


def extract_frontier_curves(
    summary_df: Any,
    threshold: float = _SYNC_THRESH,
    family: str | None = None,
) -> dict[float, list[dict[str, Any]]]:
    """Extract observed rho*(s_in, alpha) points and both censoring flags."""
    df = summary_df
    if family is not None:
        df = df[df["family_name"] == family]
    if len(df) == 0:
        return {}

    rho_max = float(df["rho_target"].astype(float).max())
    curves: dict[float, list[dict[str, Any]]] = {}
    for alpha in sorted(df["alpha"].astype(float).unique()):
        sub_alpha = _isclose_filter(df, "alpha", alpha)
        points: list[dict[str, Any]] = []
        for s_in in sorted(sub_alpha["s_in"].astype(float).unique()):
            group = _isclose_filter(sub_alpha, "s_in", s_in)
            valid = group[
                group["fraction_synchronized_mean"].astype(float) >= threshold
            ]
            if len(valid) == 0:
                continue
            rho_star = float(valid["rho_target"].astype(float).max())
            temporal_extension = temporal_extension_rhos(group, rho_star)
            temporal = bool(temporal_extension)
            temporal_rho = temporal_extension[-1] if temporal_extension else None
            points.append({
                "alpha": float(alpha),
                "input_scaling": float(s_in),
                "rho_star": rho_star,
                "spatially_truncated": detect_spatial_truncation(rho_star, rho_max),
                "temporally_truncated_next": temporal,
                "temporal_candidate_rho": temporal_rho,
                "temporal_extension_rhos": temporal_extension,
            })
        if points:
            curves[float(alpha)] = points
    return curves


def build_frontier_surface(
    frontier_curves: dict[float, list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a visual interpolation mesh from observed frontier curves."""
    if not frontier_curves:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty.copy(), empty.copy()

    alphas = sorted(frontier_curves)
    sin_values = sorted({
        float(point["input_scaling"])
        for points in frontier_curves.values()
        for point in points
    })
    X = np.full((len(alphas), len(sin_values)), np.nan)
    Y = np.tile(np.asarray(sin_values, dtype=float), (len(alphas), 1))
    Z = np.tile(np.asarray(alphas, dtype=float)[:, None], (1, len(sin_values)))

    for alpha_idx, alpha in enumerate(alphas):
        by_sin = {
            float(point["input_scaling"]): float(point["rho_star"])
            for point in frontier_curves[alpha]
        }
        for sin_idx, s_in in enumerate(sin_values):
            if s_in in by_sin:
                X[alpha_idx, sin_idx] = by_sin[s_in]
    return X, Y, Z


def _region_3d_spec(region: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy 2-D region dictionaries into a 3-D definition."""
    name = str(region["name"])
    default = _THEORETICAL_REGIONS.get(name, {})
    return {
        "name": name,
        "rho": tuple(region.get("rho", region.get("x", default.get("rho", (0.0, 0.0))))),
        "input_scaling": tuple(
            region.get("input_scaling", region.get("y", default.get("input_scaling", (0.0, 0.0))))
        ),
        "leak_rate": tuple(
            region.get("leak_rate", region.get("alpha", default.get("leak_rate", (1.0, 1.0))))
        ),
    }


def _overlay_theoretical_regions_2d(
    ax: Any,
    regions: list[dict[str, Any]],
    x_ticks: list[float],
    y_ticks: list[float],
) -> None:
    """Overlay R1-R3 as slice regions and R4 as a low-leak projection."""
    from matplotlib.patches import Rectangle

    colors = {"R1": "#1f77b4", "R2": "#2ca02c", "R3": "#ff7f0e", "R4": "#9467bd"}
    for region in regions:
        spec = _region_3d_spec(region)
        name = spec["name"]
        x_lo, x_hi = spec["rho"]
        y_lo, y_hi = spec["input_scaling"]
        x0 = categorical_index(x_ticks, x_lo) - 0.5
        x1 = categorical_index(x_ticks, x_hi) + 0.5
        y0 = categorical_index(y_ticks, y_lo) - 0.5
        y1 = categorical_index(y_ticks, y_hi) + 0.5
        is_projection = name == "R4"
        color = colors.get(name, "#555555")
        alpha = 0.42 if is_projection else 0.82
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
            linestyle="--" if is_projection else "-",
            alpha=alpha,
            zorder=4,
        )
        ax.add_patch(rect)
        label = "R4 proj." if is_projection else name
        ax.text(
            x0 + 0.02 * (x1 - x0),
            y1 - 0.04 * (y1 - y0),
            label,
            fontsize=7,
            color=color,
            alpha=0.7 if is_projection else 1.0,
            va="top",
            ha="left",
            zorder=6,
        )


# ---------------------------------------------------------------------------
# F7 helper — translucent 3-D box
# ---------------------------------------------------------------------------

def _draw_box_3d(
    ax: Any,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    color: str,
    alpha_face: float = 0.12,
    label: str | None = None,
) -> None:
    """Add a translucent axis-aligned box to a 3-D axes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    verts = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    poly = Poly3DCollection(
        verts, alpha=alpha_face,
        facecolor=color, edgecolor=color, linewidths=0.7,
    )
    ax.add_collection3d(poly)

    edges = [
        ((x0, y0, z0), (x1, y0, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)),
        ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)),
        ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)),
        ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)),
        ((x0, y1, z0), (x0, y1, z1)),
    ]
    for start, end in edges:
        ax.plot(
            [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
            color=color, lw=0.9, alpha=0.95,
        )
    if label:
        ax.text(x0, y1, z1, label, fontsize=7, color=color, zorder=10)


def _draw_rect_3d(
    ax: Any,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z: float,
    color: str,
    alpha_face: float = 0.12,
    label: str | None = None,
) -> None:
    """Add a translucent rectangle on a fixed-z plane to a 3-D axes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x0, x1 = x_range
    y0, y1 = y_range
    verts = [[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]]
    poly = Poly3DCollection(
        verts, alpha=alpha_face,
        facecolor=color, edgecolor=color, linewidths=0.7,
    )
    ax.add_collection3d(poly)
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5,
            color=color, lw=0.9, alpha=0.95)
    if label:
        ax.text(x0, y1, z, label, fontsize=7, color=color, zorder=10)


def _draw_experimental_grid_mesh(
    ax: Any,
    points: list[dict[str, Any]],
    color: str,
    grid_name: str,
) -> None:
    """Connect adjacent points within one Cartesian experimental grid."""
    coordinates = ("spectral_radius", "input_scaling", "leak_rate")
    for varying in coordinates:
        fixed = tuple(name for name in coordinates if name != varying)
        grouped: dict[tuple[float, float], list[dict[str, float]]] = {}
        for point in points:
            key = (float(point[fixed[0]]), float(point[fixed[1]]))
            grouped.setdefault(key, []).append(point)

        for line_points in grouped.values():
            ordered = sorted(line_points, key=lambda point: float(point[varying]))
            if len(ordered) < 2:
                continue
            line = ax.plot(
                [point["spectral_radius"] for point in ordered],
                [point["input_scaling"] for point in ordered],
                [point["leak_rate"] for point in ordered],
                color=color,
                lw=0.5,
                alpha=0.20,
                zorder=4,
            )[0]
            line.set_gid(f"experimental-grid-mesh-{grid_name}")


# ---------------------------------------------------------------------------
# F1 — frac_sync heatmap (ρ, s_in) @ α_primary + frontier
# ---------------------------------------------------------------------------

def _f1_frac_sync(
    df: Any, out_dir: Path, family: str, alpha_primary: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    fixed = _fixed_alpha(family, alpha_primary)
    grid, xs, ys, _ = pivot_plane(df, "fraction_synchronized_mean", "rho_target", "s_in", fixed)
    fp = frontier_per_sin(df, "rho_target", "s_in", "fraction_synchronized_mean",
                          _SYNC_THRESH, fixed)

    fig, ax = plt.subplots(figsize=(7, 4))
    pcm = heatmap_plane(ax, grid, xs, ys, CMAPS["sequential_cool"], 0.0, 1.0)
    overlay_frontier_per_y(ax, xs, ys, fp)
    add_colorbar(fig, ax, pcm, _LAB_FRAC)
    categorical_vline(ax, xs, 1.0, color="k", lw=0.8, ls=":", zorder=6)
    ax.set_xlabel(_LAB_RHO)
    ax.set_ylabel(_LAB_SIN)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "F1_frac_sync")


# ---------------------------------------------------------------------------
# F2 — washout heatmap (log color, mask frac<1) + frontier + regions
# ---------------------------------------------------------------------------

def _f2_washout_heatmap(
    df: Any, out_dir: Path, family: str, alpha_primary: float,
    regions: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    fixed = _fixed_alpha(family, alpha_primary)
    grid, xs, ys, _ = pivot_plane(df, "sync_time_mean_mean", "rho_target", "s_in", fixed)
    smask = _sync_mask(df, "rho_target", "s_in", fixed)
    log_g = _log_grid(grid)
    vmin = float(np.nanmin(log_g)) if not np.all(np.isnan(log_g)) else 0.0
    vmax = float(np.nanmax(log_g)) if not np.all(np.isnan(log_g)) else 1.0
    fp = frontier_per_sin(df, "rho_target", "s_in", "fraction_synchronized_mean",
                          _SYNC_THRESH, fixed)

    fig, ax = plt.subplots(figsize=(7, 4))
    pcm = heatmap_plane(ax, log_g, xs, ys, CMAPS["sequential_warm"], vmin, vmax, mask=smask)
    overlay_frontier_per_y(ax, xs, ys, fp)
    _overlay_theoretical_regions_2d(ax, regions, xs, ys)
    add_colorbar(fig, ax, pcm, _LAB_LOGW)
    categorical_vline(ax, xs, 1.0, color="k", lw=0.8, ls=":", zorder=6)
    ax.set_xlabel(_LAB_RHO)
    ax.set_ylabel(_LAB_SIN)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "F2_washout_heatmap")


# ---------------------------------------------------------------------------
# F3 — amplitud media ⟨|x|⟩ heatmap + frontier + regions
# ---------------------------------------------------------------------------

def _f3_sat_heatmap(
    df: Any, out_dir: Path, family: str, alpha_primary: float,
    regions: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    fixed = _fixed_alpha(family, alpha_primary)
    grid, xs, ys, _ = pivot_plane(df, "saturation_mean_mean", "rho_target", "s_in", fixed)
    fp = frontier_per_sin(df, "rho_target", "s_in", "fraction_synchronized_mean",
                          _SYNC_THRESH, fixed)

    # Use data-driven vmax so the gradient is visible
    vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
    vmax = max(vmax, 1e-3)   # guard against all-zero data in tests

    fig, ax = plt.subplots(figsize=(7, 4))
    pcm = heatmap_plane(ax, grid, xs, ys, CMAPS["sequential_warm"], 0.0, vmax)
    overlay_frontier_per_y(ax, xs, ys, fp)
    _overlay_theoretical_regions_2d(ax, regions, xs, ys)
    add_colorbar(fig, ax, pcm, _LAB_AMP)
    categorical_vline(ax, xs, 1.0, color="k", lw=0.8, ls=":", zorder=6)
    ax.set_xlabel(_LAB_RHO)
    ax.set_ylabel(_LAB_SIN)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "F3_sat_heatmap")


# ---------------------------------------------------------------------------
# F4 — ρ*(s_in) one line per α + ρ=1 reference; mark truncated
# ---------------------------------------------------------------------------

def _f4_rho_star_curves(df: Any, out_dir: Path, family: str) -> list[Path]:
    import matplotlib.pyplot as plt

    alpha_vals = sorted(df["alpha"].unique().astype(float))
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, alpha_val in enumerate(alpha_vals):
        fixed = _fixed_alpha(family, alpha_val)
        fp = frontier_per_x(df, "s_in", "rho_target", "fraction_synchronized_mean",
                            _SYNC_THRESH, fixed)
        if not fp:
            continue

        sin_vals = sorted(fp)
        rho_stars = [fp[s] for s in sin_vals]
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(sin_vals, rho_stars, color=color, marker="o", ms=4,
                lw=1.5, label=rf"$\alpha={alpha_val}$")

        # Mark truncated: rho just above rho* where nonsync_desc >= _TRUNC_THRESH
        sub_f = _isclose_filter(df[df["family_name"] == family], "alpha", alpha_val)
        for s_in_val, rho_star_val in fp.items():
            above = sub_f[
                np.isclose(sub_f["s_in"].astype(float).to_numpy(), s_in_val) &
                (sub_f["rho_target"].astype(float) > rho_star_val)
            ].sort_values("rho_target")
            if len(above) > 0:
                nfd_f = _nfd_float(above.iloc[0])
                if np.isfinite(nfd_f) and nfd_f >= _TRUNC_THRESH:
                    ax.scatter(s_in_val, rho_star_val, s=60, marker="^",
                               color=color, zorder=7, edgecolors="k", lw=0.5)

    ax.axhline(1.0, color="k", lw=0.8, ls=":", label=r"$\rho=1$", zorder=3)
    ax.set_xlabel(_LAB_SIN)
    ax.set_ylabel(r"$\rho^{*}$ (máx $\rho$ sincronizado)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_dir, "F4_rho_star_curves")


# ---------------------------------------------------------------------------
# F5 — washout(ρ) per s_in @ α_primary, y log
#       series split at ρ* + lower panel frac_sync(ρ)
# ---------------------------------------------------------------------------

def _f5_washout_lines(
    df: Any, out_dir: Path, family: str, alpha_primary: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    sub = df[df["family_name"] == family].copy()
    sub = _isclose_filter(sub, "alpha", alpha_primary)
    sin_groups = sorted(sub["s_in"].astype(float).unique())

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )

    for idx, sin_val in enumerate(sin_groups):
        sub_s = sub[
            np.isclose(sub["s_in"].astype(float).to_numpy(), sin_val)
        ].sort_values("rho_target")
        color = PALETTE[idx % len(PALETTE)]
        label = rf"$s_{{\mathrm{{in}}}}={sin_val:.2g}$"

        # ρ* for this s_in
        valid_rows = sub_s[sub_s["fraction_synchronized_mean"].astype(float) >= _SYNC_THRESH]
        rho_star = (float(valid_rows["rho_target"].astype(float).max())
                    if len(valid_rows) > 0 else float("-inf"))

        solid = sub_s[
            (sub_s["rho_target"].astype(float) <= rho_star) &
            sub_s["sync_time_mean_mean"].notna()
        ]
        dashed = sub_s[
            (sub_s["rho_target"].astype(float) > rho_star) &
            sub_s["sync_time_mean_mean"].notna()
        ]

        kw: dict[str, Any] = dict(color=color, ms=3, lw=1.5)
        if len(solid) > 0:
            ax_top.plot(solid["rho_target"].astype(float).to_numpy(),
                        solid["sync_time_mean_mean"].astype(float).to_numpy(),
                        ls="-", marker="o", **kw, label=label)
        if len(dashed) > 0:
            dash_x = dashed["rho_target"].astype(float).to_numpy()
            dash_y = dashed["sync_time_mean_mean"].astype(float).to_numpy()
            if len(solid) > 0:
                dash_x = np.concatenate([
                    solid["rho_target"].astype(float).to_numpy()[-1:],
                    dash_x,
                ])
                dash_y = np.concatenate([
                    solid["sync_time_mean_mean"].astype(float).to_numpy()[-1:],
                    dash_y,
                ])
            ax_top.plot(dash_x, dash_y, ls="--", marker="o", mfc="none",
                        alpha=0.5, **kw, label=label if len(solid) == 0 else None)

        # Bottom panel: frac_sync(ρ)
        ax_bot.plot(sub_s["rho_target"].astype(float).to_numpy(),
                    sub_s["fraction_synchronized_mean"].astype(float).to_numpy(),
                    ls="-", marker="o", ms=2, lw=1.2, color=color)

    ax_top.set_yscale("log")
    ax_top.axvline(1.0, color="k", lw=0.8, ls=":", zorder=3)
    ax_top.set_ylabel(_LAB_WASHOUT)
    ax_top.legend(fontsize=7, loc="upper left")

    ax_bot.axvline(1.0, color="k", lw=0.8, ls=":", zorder=3)
    ax_bot.axhline(_SYNC_THRESH, color="gray", lw=0.6, ls=":", zorder=3)
    ax_bot.set_xlabel(_LAB_RHO)
    ax_bot.set_ylabel("frac. sync.")
    ax_bot.set_ylim(-0.05, 1.05)

    return save_figure(fig, out_dir, "F5_washout_lines")


# ---------------------------------------------------------------------------
# F6 — cortes leak: heatmaps washout(ρ,α) y amplitud(ρ,α) a s_in fijo
# ---------------------------------------------------------------------------

def _f6_leak_cuts(df: Any, out_dir: Path, family: str) -> list[Path]:
    import matplotlib.pyplot as plt

    available_sin = sorted(df["s_in"].astype(float).unique())
    sin_cuts = [min(available_sin, key=lambda x, t=t: abs(x - t)) for t in (0.1, 0.8)]

    # (metric_col, label, do_log, fixed_vmax_or_None)
    metrics = [
        ("sync_time_mean_mean", _LAB_LOGW, True, None),
        ("saturation_mean_mean", _LAB_AMP, False, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    for row_idx, sin_val in enumerate(sin_cuts):
        fixed: dict[str, Any] = {"family_name": family, "s_in": sin_val}

        for col_idx, (metric_col, metric_label, do_log, _fixed_vmax) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            grid, xs, ys, _ = pivot_plane(df, metric_col, "rho_target", "alpha", fixed)

            if do_log:
                plot_grid = _log_grid(grid)
                cb_label = metric_label
            else:
                plot_grid = grid
                cb_label = metric_label

            vmin = float(np.nanmin(plot_grid)) if not np.all(np.isnan(plot_grid)) else 0.0
            vmax = float(np.nanmax(plot_grid)) if not np.all(np.isnan(plot_grid)) else 1.0
            vmax = max(vmax, vmin + 1e-6)   # guard against flat data in tests

            pcm = heatmap_plane(ax, plot_grid, xs, ys, CMAPS["sequential_warm"], vmin, vmax)
            add_colorbar(fig, ax, pcm, cb_label)
            categorical_vline(ax, xs, 1.0, color="k", lw=0.8, ls=":", zorder=6)
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(r"$\alpha$")
            metric_short = r"$\log_{10}$(washout)" if do_log else r"amplitud $\langle|x|\rangle$"
            ax.set_title(
                rf"$s_{{\mathrm{{in}}}}={sin_val:.2f}$ — {metric_short}"
            )

    fig.tight_layout()
    return save_figure(fig, out_dir, "F6_leak_cuts")


# ---------------------------------------------------------------------------
# F7 - parameter space: R1-R4 regions and experimental points
# ---------------------------------------------------------------------------

def _f7_surface_3d(
    df: Any, out_dir: Path, family: str,
    regions: list[dict[str, Any]] = CANDIDATE_REGIONS,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sub = df[df["family_name"] == family]
    sin_vals = sorted(sub["s_in"].astype(float).unique())
    alpha_vals = sorted(sub["alpha"].astype(float).unique())
    rho_vals = sorted(sub["rho_target"].astype(float).unique())
    alpha_min, alpha_max = min(alpha_vals), max(alpha_vals)
    rho_max_curves = extract_frontier_curves(sub, family=family)
    surface_x, surface_y, surface_z = build_frontier_surface(rho_max_curves)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Layer 1: conceptual dense regions.
    region_colors = {
        "R1": "#1f77b4",
        "R2": "#2ca02c",
        "R3": "#ff7f0e",
        "R4": "#9467bd",
    }
    for r in regions:
        spec = _region_3d_spec(r)
        color = region_colors.get(str(spec["name"]), "#555555")
        _draw_box_3d(
            ax,
            x_range=spec["rho"],
            y_range=spec["input_scaling"],
            z_range=spec["leak_rate"],
            color=color,
            alpha_face=0.055,
            label=spec["name"],
        )

    # Layer 2: the finite grids used by the later performance sweep.
    grid_styles = {
        "A": ("o", "#1f77b4", "A (rejilla, R1/R2)"),
        "B1": ("s", "#ff7f0e", "B1 (rejilla, R3)"),
        "B2": ("^", "#d62728", "B2 (rejilla, R3 extremo)"),
        "C": ("D", "#9467bd", "C (rejilla, R4)"),
    }
    grids = build_experimental_grids()
    for grid_name, points in grids.items():
        marker, color, _ = grid_styles[grid_name]
        _draw_experimental_grid_mesh(ax, points, color, grid_name)
        ax.scatter(
            [p["spectral_radius"] for p in points],
            [p["input_scaling"] for p in points],
            [p["leak_rate"] for p in points],
            marker=marker,
            s=16,
            c=color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.78,
            depthshade=False,
            zorder=5,
        )

    # Layer 3: colored approximation of the rho_max manifold and its mesh.
    if surface_x.size and surface_x.shape[0] >= 2 and surface_x.shape[1] >= 2:
        manifold = ax.plot_surface(
            surface_x,
            surface_y,
            surface_z,
            color="#76b7e5",
            alpha=0.14,
            edgecolor="none",
            shade=False,
            antialiased=True,
            zorder=3,
        )
        manifold.set_gid("rho-max-manifold")
        manifold_mesh = ax.plot_wireframe(
            surface_x,
            surface_y,
            surface_z,
            rstride=1,
            cstride=1,
            color="#164a73",
            linewidth=0.65,
            alpha=0.30,
            zorder=4,
        )
        manifold_mesh.set_gid("rho-max-manifold-mesh")

    # Layer 4: observed rho_max curves, dashed only for temporal truncation.
    rho_colors = plt.get_cmap("Blues")(
        np.linspace(0.52, 0.92, max(1, len(rho_max_curves)))
    )
    rho_handles: list[Any] = []
    for idx, (alpha_val, points) in enumerate(sorted(rho_max_curves.items())):
        color = rho_colors[idx]
        ordered = sorted(points, key=lambda point: float(point["input_scaling"]))
        for left, right in zip(ordered, ordered[1:]):
            temporal_segment = bool(
                left["temporally_truncated_next"]
                or right["temporally_truncated_next"]
            )
            segment = ax.plot(
                [float(left["rho_star"]), float(right["rho_star"])],
                [float(left["input_scaling"]), float(right["input_scaling"])],
                [float(alpha_val), float(alpha_val)],
                color=color,
                lw=2.0,
                ls="--" if temporal_segment else "-",
                alpha=0.72 if temporal_segment else 0.95,
                zorder=8,
            )[0]
            segment.set_gid(
                "rho-max-segment-temporal"
                if temporal_segment
                else "rho-max-segment-observed"
            )

        normal = [
            point for point in ordered
            if not (
                point["temporally_truncated_next"]
                or point["spatially_truncated"]
            )
        ]
        temporal = [
            point for point in ordered
            if point["temporally_truncated_next"]
            and not point["spatially_truncated"]
        ]
        spatial = [point for point in ordered if point["spatially_truncated"]]
        if normal:
            normal_points = ax.plot(
                [float(point["rho_star"]) for point in normal],
                [float(point["input_scaling"]) for point in normal],
                [float(alpha_val)] * len(normal),
                ls="none",
                marker="o",
                ms=4.2,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=9,
            )[0]
            normal_points.set_gid("rho-max-point-observed")
        if temporal:
            temporal_points = ax.plot(
                [float(point["rho_star"]) for point in temporal],
                [float(point["input_scaling"]) for point in temporal],
                [float(alpha_val)] * len(temporal),
                ls="none",
                marker="o",
                ms=5.0,
                color=color,
                markerfacecolor="none",
                markeredgewidth=1.1,
                zorder=9,
            )[0]
            temporal_points.set_gid("rho-max-point-temporal")
        if spatial:
            spatial_points = ax.plot(
                [float(point["rho_star"]) for point in spatial],
                [float(point["input_scaling"]) for point in spatial],
                [float(alpha_val)] * len(spatial),
                ls="none",
                marker=">",
                ms=6.0,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.45,
                zorder=10,
            )[0]
            spatial_points.set_gid("rho-max-point-spatial")

        rho_handles.append(Line2D(
            [0], [0], color=color, lw=2.0, marker="o", ms=4,
            label=rf"$\alpha={alpha_val:g}$",
        ))

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$s_{\mathrm{in}}$")
    ax.set_zlabel(r"$\alpha$")
    ax.set_xlim(min(rho_vals), max(rho_vals))
    ax.set_ylim(min(sin_vals), max(sin_vals))
    ax.set_zlim(min(0.1, alpha_min), max(1.0, alpha_max))
    ax.view_init(elev=24, azim=-62)

    rho_legend = ax.legend(
        handles=rho_handles,
        title=r"Curvas por $\alpha$",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
    )
    ax.add_artist(rho_legend)

    grid_handles: list[Any] = []
    grid_handles.extend([
        Line2D([0], [0], color="#2171b5", lw=2.0, ls="-",
               label=r"$\rho_{\max}$ observado"),
        Line2D([0], [0], color="#2171b5", lw=2.0, ls="--",
               marker="o", markerfacecolor="none",
               label="truncamiento temporal"),
        Line2D([0], [0], color="#2171b5", lw=0, marker=">",
               markeredgecolor="black",
               label="truncamiento espacial"),
        Patch(facecolor="#76b7e5", edgecolor="#164a73", alpha=0.22,
              label=r"variedad aproximada de $\rho_{\max}$"),
    ])
    for grid_name in ("A", "B1", "B2", "C"):
        marker, color, label = grid_styles[grid_name]
        grid_handles.append(Line2D(
            [0], [0], marker=marker, color="none", markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.4, markersize=5,
            label=label,
        ))
    ax.legend(
        handles=grid_handles,
        title="Capas",
        fontsize=7,
        title_fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.00, 0.42),
    )
    fig.subplots_adjust(left=0.02, right=0.74, bottom=0.02, top=0.93)
    return save_figure(fig, out_dir, "F7_surface_3d")


# ---------------------------------------------------------------------------
# F8 — 3 trazas d_curve: sub-crítica, frontera-truncada, super-crítica
# ---------------------------------------------------------------------------

def _f8_dcurves(
    df: Any, out_dir: Path, runs_dir: Path, family: str, alpha_primary: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    sub = df[df["family_name"] == family].copy()
    sub = _isclose_filter(sub, "alpha", alpha_primary)

    sub_crit = sub[
        (sub["rho_target"].astype(float) < 0.9) &
        (sub["fraction_synchronized_mean"].astype(float) >= _SYNC_THRESH)
    ]
    near_trunc = sub[sub["nonsync_fraction_descending"].astype(float) >= _TRUNC_THRESH]
    super_crit = sub[
        (sub["rho_target"].astype(float) > 1.2) &
        (sub["fraction_synchronized_mean"].astype(float) < 0.1)
    ]

    candidates: list[tuple[str, float, dict[str, Any]]] = []
    for label, frame in [
        ("sub-crítica", sub_crit),
        ("frontera-truncada", near_trunc),
        ("super-crítica", super_crit),
    ]:
        if len(frame) == 0:
            continue
        row = frame.iloc[0]
        curves = load_point_curves(runs_dir, {
            "family_name": family,
            "alpha": float(row["alpha"]),
            "s_in": float(row["s_in"]),
            "rho_target": float(row["rho_target"]),
        })
        if curves:
            candidates.append((label, float(row["rho_target"]), curves[0]))

    if not candidates:
        return []

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, (label, rho, cdata) in enumerate(candidates):
        d_sub = cdata.get("d_curve_mean_sub", [])
        if d_sub:
            ax.plot(list(range(len(d_sub))), d_sub,
                    label=rf"{label} ($\rho={rho:.2f}$)",
                    color=PALETTE[idx % len(PALETTE)], lw=1.5)

    ax.axhline(1e-3, color="k", lw=0.8, ls=":", label=r"$\varepsilon=10^{-3}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"paso ($\times 10$, submuestreo)")
    ax.set_ylabel(r"$d(t)$ relativa (log)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return save_figure(fig, out_dir, "F8_dcurves")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_frontier_figures(
    summary_csv: str | Path,
    out_dir: str | Path,
    runs_dir: str | Path | None = None,
    regions: list[dict[str, Any]] = CANDIDATE_REGIONS,
    alpha_primary: float = 1.0,
) -> list[Path]:
    """
    Generate and save all frontier figures (F1–F8).

    Parameters
    ----------
    summary_csv  : path to a frontier summary CSV.
    out_dir      : directory where figures are written.
    runs_dir     : directory of per-point JSONs (needed for F8 only).
    regions      : list of candidate region dicts for overlay.
    alpha_primary: leak rate used as the primary slice for F1–F3, F5.

    Returns
    -------
    Sorted list of all written file paths.
    """
    apply_style()
    df = load_summary(summary_csv)
    out_dir = Path(out_dir)
    family = _family_name(df)

    saved: list[Path] = []
    saved += _f1_frac_sync(df, out_dir, family, alpha_primary)
    saved += _f2_washout_heatmap(df, out_dir, family, alpha_primary, regions)
    saved += _f3_sat_heatmap(df, out_dir, family, alpha_primary, regions)
    saved += _f4_rho_star_curves(df, out_dir, family)
    saved += _f5_washout_lines(df, out_dir, family, alpha_primary)
    saved += _f6_leak_cuts(df, out_dir, family)
    saved += _f7_surface_3d(df, out_dir, family, regions)
    if runs_dir is not None:
        saved += _f8_dcurves(df, out_dir, Path(runs_dir), family, alpha_primary)

    return sorted(saved)
