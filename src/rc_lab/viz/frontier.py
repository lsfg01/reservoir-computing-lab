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
    categorical_vline,
    heatmap_plane,
    overlay_frontier_per_y,
    overlay_regions,
    surface_plane,  # kept in public API; not used by F7 after rewrite
)
from rc_lab.viz.style import CMAPS, PALETTE, apply_style, save_figure

# Default candidate regions (refined post-campaign)
CANDIDATE_REGIONS: list[dict[str, Any]] = [
    {"name": "R1", "x": (0.60, 0.90), "y": (0.05, 0.20)},
    {"name": "R2", "x": (0.90, 1.00), "y": (0.05, 0.20)},
    {"name": "R3", "x": (1.00, 1.40), "y": (0.40, 1.50)},
    {"name": "R4", "x": (0.80, 1.10), "y": (0.05, 0.40)},
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


def _alpha_title(alpha: float) -> str:
    """Produce e.g. '($\\alpha{=}1.0$)' for figure titles."""
    return rf"($\alpha{{=}}{alpha}$)"


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
    ax.set_title(rf"F1 — Fracción sincronizada {_alpha_title(alpha_primary)}")
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
    overlay_regions(ax, regions, xs, ys)
    add_colorbar(fig, ax, pcm, _LAB_LOGW)
    categorical_vline(ax, xs, 1.0, color="k", lw=0.8, ls=":", zorder=6)
    ax.set_xlabel(_LAB_RHO)
    ax.set_ylabel(_LAB_SIN)
    ax.set_title(rf"F2 — Washout empírico $(\log_{{10}})$ {_alpha_title(alpha_primary)}")
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
    overlay_regions(ax, regions, xs, ys)
    add_colorbar(fig, ax, pcm, _LAB_AMP)
    categorical_vline(ax, xs, 1.0, color="k", lw=0.8, ls=":", zorder=6)
    ax.set_xlabel(_LAB_RHO)
    ax.set_ylabel(_LAB_SIN)
    ax.set_title(rf"F3 — Amplitud media $\langle|x|\rangle$ {_alpha_title(alpha_primary)}")
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
    ax.set_title(r"F4 — Frontera $\rho^{*}(s_{\mathrm{in}})$ por $\alpha$  [$\blacktriangle$ = truncado]")
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
    ax_top.set_title(
        rf"F5 — Washout$(\rho)$ por $s_{{\mathrm{{in}}}}$ {_alpha_title(alpha_primary)}"
        "\n" r"(— sync.; -- censurado)"
    )
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
    fig.suptitle(
        r"F6 — Cortes de leak: washout y amplitud $\langle|x|\rangle$ por $(\rho, \alpha)$",
        fontsize=10,
    )

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
# F7 — parameter space (ρ, s_in, α): R1–R4 boxes + ρ* scatter
# ---------------------------------------------------------------------------

def _f7_surface_3d(
    df: Any, out_dir: Path, family: str,
    regions: list[dict[str, Any]] = CANDIDATE_REGIONS,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sub = df[df["family_name"] == family]
    sin_vals = sorted(sub["s_in"].astype(float).unique())
    alpha_vals = sorted(sub["alpha"].astype(float).unique())
    rho_vals = sorted(sub["rho_target"].astype(float).unique())
    alpha_min, alpha_max = min(alpha_vals), max(alpha_vals)

    # Curves are computed below after the regions are drawn.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw R1-R4 with their real alpha extent.
    region_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for idx, r in enumerate(regions):
        color = region_colors[idx % len(region_colors)]
        if r["name"] == "R4":
            _draw_box_3d(
                ax,
                x_range=r["x"],
                y_range=r["y"],
                z_range=(0.3, 0.5),
                color=color,
                alpha_face=0.12,
                label=r["name"],
            )
        else:
            _draw_rect_3d(
                ax,
                x_range=r["x"],
                y_range=r["y"],
                z=1.0,
                color=color,
                alpha_face=0.12,
                label=r["name"],
            )

    trunc_legend_added = False
    for idx, alpha_val in enumerate(alpha_vals):
        fixed = _fixed_alpha(family, alpha_val)
        fp = frontier_per_sin(df, "rho_target", "s_in", "fraction_synchronized_mean",
                              _SYNC_THRESH, fixed)
        if not fp:
            continue

        curve_s = sorted(fp)
        curve_rho = [fp[s_in_val] for s_in_val in curve_s]
        curve_trunc: list[bool] = []
        sub_a = _isclose_filter(sub, "alpha", alpha_val)

        for s_in_val, rho_star_val in zip(curve_s, curve_rho):
            sub_s = sub_a[np.isclose(sub_a["s_in"].astype(float).to_numpy(), s_in_val)]
            valid = sub_s[sub_s["fraction_synchronized_mean"].astype(float) >= _SYNC_THRESH]
            above = sub_s[
                sub_s["rho_target"].astype(float) > rho_star_val
            ].sort_values("rho_target")
            is_trunc = len(valid) == 0
            if len(above) > 0:
                nfd_f = _nfd_float(above.iloc[0])
                if np.isfinite(nfd_f) and nfd_f >= _TRUNC_THRESH:
                    is_trunc = True
            curve_trunc.append(is_trunc)

        color = PALETTE[idx % len(PALETTE)]
        ax.plot([], [], [], color=color, lw=1.8, label=rf"$\alpha={alpha_val:g}$")
        for seg_idx in range(len(curve_s) - 1):
            seg_trunc = curve_trunc[seg_idx] or curve_trunc[seg_idx + 1]
            ax.plot(
                [curve_rho[seg_idx], curve_rho[seg_idx + 1]],
                [curve_s[seg_idx], curve_s[seg_idx + 1]],
                [alpha_val, alpha_val],
                color=color,
                lw=1.8,
                ls="--" if seg_trunc else "-",
                alpha=0.7 if seg_trunc else 1.0,
                zorder=6,
            )

        normal_pts = [
            (rho, s_in_val, alpha_val)
            for rho, s_in_val, is_trunc in zip(curve_rho, curve_s, curve_trunc)
            if not is_trunc
        ]
        trunc_pts = [
            (rho, s_in_val, alpha_val)
            for rho, s_in_val, is_trunc in zip(curve_rho, curve_s, curve_trunc)
            if is_trunc
        ]
        if normal_pts:
            xs, ys, zs = zip(*normal_pts)
            ax.scatter(xs, ys, zs, s=24, c=color, marker="o", depthshade=True,
                       zorder=7)
        if trunc_pts:
            xs, ys, zs = zip(*trunc_pts)
            ax.scatter(
                xs, ys, zs, s=34, c="none", edgecolors=color,
                linewidths=1.2, marker="o",
                label="truncado" if not trunc_legend_added else None,
                depthshade=False, zorder=8,
            )
            trunc_legend_added = True

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$s_{\mathrm{in}}$")
    ax.set_zlabel(r"$\alpha$")
    ax.set_xlim(min(rho_vals), max(rho_vals))
    ax.set_ylim(min(sin_vals), max(sin_vals))
    ax.set_zlim(min(0.3, alpha_min), max(1.0, alpha_max))
    ax.view_init(elev=22, azim=-60)
    ax.set_title(
        r"F7 — Regiones y frontera $\rho^{*}$ en $(\rho, s_{\mathrm{in}}, \alpha)$"
    )
    ax.legend(fontsize=7, loc="upper left")
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
    ax.set_title(r"F8 — Curvas $d(t)$: sincronización vs. divergencia")
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
