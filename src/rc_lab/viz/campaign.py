"""campaign.py — Reproducible figures and LaTeX tables for final campaign results.

This module is intentionally post-hoc: it consumes persisted campaign summaries
(`summary.csv` for the multitask sweep and `comparison_summary.csv` for design)
and produces manuscript-ready artifacts without depending on the runners.

Sweep batch (regenerated):
  - F_sweep_error_vs_rho      curvas de descenso de error vs rho por s_in (S1)
  - F_sweep_task_heatmaps     planos rho x s_in por tarea (S2)
  - F_sweep_aggregate_heatmaps  meseta de compromiso (rank agregado)
  - F_sweep_tradeoff_mc_mg / _mc_narma  frontera memoria <-> no-linealidad (S3)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from rc_lab.viz.io import pivot_plane
from rc_lab.viz.primitives import add_colorbar, categorical_vline, heatmap_display_grid, heatmap_plane
from rc_lab.viz.style import CMAPS, PALETTE, apply_style, save_figure
from rc_lab.viz.tables import save_table_latex


# ---------------------------------------------------------------------------
# Constants / labels
# ---------------------------------------------------------------------------

FAMILY_ORDER = [
    "random_sparse_baseline",
    "cycle_scr",
    "cycle_jump_j3",
    "cycle_jump_j7",
    "nonnormal_chain_g0_1",
    "nonnormal_chain_g0_3",
    "nonnormal_chain_g0_6",
    "multiscale_two_random",
    "multiscale_three_random",
]

FAMILY_LABELS = {
    "random_sparse_baseline": "random sparse",
    "cycle_scr": "SCR",
    "cycle_jump_j3": "CRJ j=3",
    "cycle_jump_j7": "CRJ j=7",
    "nonnormal_chain_g0_1": "cadena g=0.1",
    "nonnormal_chain_g0_3": "cadena g=0.3",
    "nonnormal_chain_g0_6": "cadena g=0.6",
    "multiscale_two_random": "multiescala 2",
    "multiscale_three_random": "multiescala 3",
}

TASK_LABELS = {
    "narma10": "NARMA-10",
    "mackey_glass": "Mackey–Glass",
    "memory_capacity": "MC",
    "aggregate": "agregado",
}

TASK_RANK_COLS = {
    "narma10": "global_rank_narma10",
    "mackey_glass": "global_rank_mg",
    "memory_capacity": "global_rank_mc",
}

TASK_METRIC_COLS = {
    "narma10": "narma10_val_nmse_mean",
    "mackey_glass": "mg_val_nmse_mean",
    "memory_capacity": "mc_total_mean",
}

# Clean, ordered palette for s_in lines (light, distinguishable; legend conveys order)
SIN_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

LAB_RHO = r"$\rho$"
LAB_SIN = r"$s_{\mathrm{in}}$"
LAB_ALPHA = r"$\alpha$"


# ---------------------------------------------------------------------------
# Loading / normalization
# ---------------------------------------------------------------------------

def load_sweep_summary(path_or_dir: str | Path) -> pd.DataFrame:
    """Load final multitask sweep summary.csv from a file or directory."""
    p = Path(path_or_dir)
    csv_path = p / "summary.csv" if p.is_dir() else p
    df = pd.read_csv(csv_path)
    _require_columns(
        df,
        [
            "config_id", "spectral_radius", "input_scaling", "leak_rate",
            "narma10_val_nmse_mean", "mg_val_nmse_mean", "mc_total_mean",
            "rank_narma10", "rank_mg", "rank_mc", "aggregate_rank",
        ],
        "sweep summary",
    )
    return df


def load_design_summary(path_or_dir: str | Path) -> pd.DataFrame:
    """Load design comparison_summary.csv from a file or directory."""
    p = Path(path_or_dir)
    csv_path = p / "comparison_summary.csv" if p.is_dir() else p
    df = pd.read_csv(csv_path)
    _require_columns(
        df,
        [
            "design_name", "reservoir_type", "config_id",
            "spectral_radius", "input_scaling", "leak_rate",
            "narma10_val_nmse_mean", "mg_val_nmse_mean", "mc_total_mean",
            "global_rank_narma10", "global_rank_mg", "global_rank_mc",
            "global_aggregate_rank",
        ],
        "design comparison summary",
    )
    return df


def _require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")


# ---------------------------------------------------------------------------
# Tables as DataFrames
# ---------------------------------------------------------------------------

def sweep_representatives(df: pd.DataFrame) -> pd.DataFrame:
    """Representative rows mapping the two-arm narrative: best compromise plus the
    single-task specialists (memory / Mackey-Glass / NARMA)."""
    selectors = [
        ("Compromiso (mejor agregado)", "aggregate_rank", "min"),
        ("Especialista memoria (MC)", "rank_mc", "min"),
        ("Especialista Mackey–Glass", "rank_mg", "min"),
        ("Especialista NARMA-10", "rank_narma10", "min"),
    ]
    rows: list[dict[str, object]] = []
    for label, col, direction in selectors:
        idx = df[col].idxmin() if direction == "min" else df[col].idxmax()
        rows.append(_sweep_display_row(df.loc[idx], label))
    return pd.DataFrame(rows)


def sweep_top_configs(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Top-n sweep configurations by aggregate rank."""
    out = df.sort_values("aggregate_rank", ascending=True).head(n).copy()
    rows = [_sweep_display_row(row, "") for _, row in out.iterrows()]
    result = pd.DataFrame(rows)
    if "rol" in result.columns:
        result = result.drop(columns=["rol"])
    return result


def design_best_by_task(df: pd.DataFrame) -> pd.DataFrame:
    """Best global design row overall and for each task."""
    selectors = [
        ("Mejor agregado", "global_aggregate_rank"),
        ("Mejor NARMA-10", "global_rank_narma10"),
        ("Mejor Mackey–Glass", "global_rank_mg"),
        ("Mejor MC", "global_rank_mc"),
    ]
    rows = [_design_display_row(df.loc[df[col].idxmin()], label) for label, col in selectors]
    return pd.DataFrame(rows)


def design_best_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Best within-family configuration, ordered by global aggregate rank."""
    idxs = df.groupby("design_name")["aggregate_rank_within_design"].idxmin()
    out = df.loc[idxs].sort_values("global_aggregate_rank", ascending=True)
    return pd.DataFrame([_design_display_row(row, "") for _, row in out.iterrows()]).drop(columns=["rol"])


def design_family_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per family: best global rank per task and best aggregate."""
    rows: list[dict[str, object]] = []
    for family in _ordered_families(df):
        sub = df[df["design_name"] == family]
        best_agg = sub.loc[sub["global_aggregate_rank"].idxmin()]
        row: dict[str, object] = {
            "familia": FAMILY_LABELS.get(family, family),
            "mejor agg.": int(best_agg["global_aggregate_rank"]),
        }
        for task, rank_col in TASK_RANK_COLS.items():
            best = sub.loc[sub[rank_col].idxmin()]
            row[f"rank {TASK_LABELS[task]}"] = int(best[rank_col])
            row[f"métrica {TASK_LABELS[task]}"] = _round(best[TASK_METRIC_COLS[task]], 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mejor agg.", ascending=True)


def save_campaign_tables(
    sweep_df: pd.DataFrame,
    design_df: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Save standard CSV + LaTeX tables for the final campaign."""
    out = Path(out_dir)
    csv_dir = out / "csv"
    tex_dir = out / "latex"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "sweep_representatives": sweep_representatives(sweep_df),
        "sweep_top12": sweep_top_configs(sweep_df, n=12),
        "design_best_by_task": design_best_by_task(design_df),
        "design_best_by_family": design_best_by_family(design_df),
        "design_family_task_summary": design_family_task_summary(design_df),
    }

    paths: dict[str, Path] = {}
    for name, table in tables.items():
        csv_path = csv_dir / f"{name}.csv"
        tex_path = tex_dir / f"{name}.tex"
        table.to_csv(csv_path, index=False)
        save_table_latex(table, tex_path, float_format="%.4g", index=False, escape=False, booktabs=True)
        paths[f"{name}_csv"] = csv_path
        paths[f"{name}_tex"] = tex_path
    return paths


def _pm(mean: object, std: object, ndigits: int) -> str | None:
    """Format 'mean ± std' compactly; None if mean is non-finite."""
    m = _round(mean, ndigits)
    if m is None:
        return None
    s = _round(std, ndigits)
    return f"{m:.{ndigits}f}" if s is None else f"{m:.{ndigits}f} ± {s:.{ndigits}f}"


def _sweep_display_row(r: pd.Series, role: str) -> dict[str, object]:
    return {
        "rol": role,
        "config": r["config_id"],
        LAB_RHO: _round(r["spectral_radius"], 3),
        LAB_SIN: _round(r["input_scaling"], 3),
        LAB_ALPHA: _round(r["leak_rate"], 3),
        "NARMA test": _pm(r.get("narma10_test_nmse_mean"), r.get("narma10_test_nmse_std"), 3),
        "MG test": _pm(r.get("mg_test_nmse_mean"), r.get("mg_test_nmse_std"), 3),
        "MC": _pm(r.get("mc_total_mean"), r.get("mc_total_std"), 2),
        "rank NARMA": int(r["rank_narma10"]),
        "rank MG": int(r["rank_mg"]),
        "rank MC": int(r["rank_mc"]),
        "rank agg.": _round(r["aggregate_rank"], 2),
    }


def _design_display_row(r: pd.Series, role: str) -> dict[str, object]:
    return {
        "rol": role,
        "familia": FAMILY_LABELS.get(str(r["design_name"]), str(r["design_name"])),
        "config": r["config_id"],
        LAB_RHO: _round(r["spectral_radius"], 3),
        LAB_SIN: _round(r["input_scaling"], 3),
        LAB_ALPHA: _round(r["leak_rate"], 3),
        "NARMA val": _round(r["narma10_val_nmse_mean"], 4),
        "MG val": _round(r["mg_val_nmse_mean"], 4),
        "MC": _round(r["mc_total_mean"], 3),
        "rank NARMA": int(r["global_rank_narma10"]),
        "rank MG": int(r["global_rank_mg"]),
        "rank MC": int(r["global_rank_mc"]),
        "rank agg.": _round(r["global_aggregate_rank"], 2),
    }


def _round(value: object, ndigits: int) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return round(x, ndigits)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_sweep_figures(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """Build the regenerated sweep figure batch."""
    apply_style()
    out = Path(out_dir)
    paths: list[Path] = []
    paths += plot_sweep_error_vs_rho(df, out, leak=1.0, formats=formats)
    paths += plot_sweep_task_heatmaps(df, out, formats=formats)
    paths += plot_sweep_aggregate_heatmaps(df, out, formats=formats)
    paths += plot_sweep_tradeoff(df, out, y_metric="mg_test_nmse_mean", name="F_sweep_tradeoff_mc_mg", formats=formats)
    paths += plot_sweep_tradeoff(df, out, y_metric="narma10_test_nmse_mean", name="F_sweep_tradeoff_mc_narma", formats=formats)
    return paths


def build_design_figures(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """Build standard reservoir-design figures (unchanged; redone in a later pass)."""
    apply_style()
    out = Path(out_dir)
    paths: list[Path] = []
    paths += plot_design_rank_heatmap(df, out, formats=formats)
    paths += plot_design_mg_vs_transient(df, out, formats=formats)
    paths += plot_design_mc_vs_henrici(df, out, formats=formats)
    paths += plot_design_delta_tradeoff(df, out, formats=formats)
    return paths


# ---------------------------------------------------------------------------
# Sweep figures (regenerated)
# ---------------------------------------------------------------------------

def plot_sweep_error_vs_rho(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    leak: float = 1.0,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """S1 — error/metric vs rho with one line per s_in, at fixed leak.

    Shows the opposite-corner structure directly: NARMA/MC are best at rho~1 and
    low drive, Mackey-Glass at high drive and rho>1. Coverage is block-dependent
    (low s_in stops near rho=1.05; high s_in starts at rho=1.0) — that gap is the
    regime structure, not missing data.
    """
    import matplotlib.pyplot as plt

    d = df[np.isclose(df["leak_rate"].astype(float), leak)]
    sins = sorted(float(v) for v in d["input_scaling"].unique())
    panels = [
        ("narma10_test_nmse_mean", "NARMA-10 NMSE", True),
        ("mg_test_nmse_mean", "Mackey-Glass NMSE", True),
        ("mc_total_mean", "Memory capacity", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (col, title, logy) in zip(axes, panels, strict=True):
        for color, s in zip(SIN_PALETTE, sins):
            sub = d[np.isclose(d["input_scaling"].astype(float), s)].sort_values("spectral_radius")
            if sub.empty:
                continue
            ax.plot(sub["spectral_radius"], sub[col], "-o", ms=4, lw=1.6, color=color, label=f"{s:g}")
        ax.axvline(1.0, ls=":", c="#999", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(LAB_RHO)
        ax.grid(True, alpha=0.25)
        if logy:
            ax.set_yscale("log")
    axes[0].legend(title=LAB_SIN, fontsize=8, ncol=2, frameon=False)
    fig.suptitle(rf"Barrido espectral por escala de entrada ($\alpha={leak:g}$)", y=1.02)
    fig.tight_layout()
    return save_figure(fig, out_dir, "F_sweep_error_vs_rho", formats=formats)


def plot_sweep_task_heatmaps(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """S2 — rho x s_in plane per task, each at a representative leak.

    Warm colormap for errors (light = low error = good); cool for MC (dark = high
    MC = good). Cell annotations use auto-contrast text; rho=1 marked.
    """
    import matplotlib.pyplot as plt

    # All panels on the SAME (rho, s_in) plane at alpha=1 for comparability.
    leak = 1.0
    specs = [
        ("narma10_test_nmse_mean", leak, "NARMA-10 NMSE", CMAPS["sequential_warm"]),
        ("mg_test_nmse_mean", leak, "Mackey-Glass NMSE", CMAPS["sequential_warm"]),
        ("mc_total_mean", leak, "Memory capacity", CMAPS["sequential_cool"]),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(15.5, 4.1), squeeze=False)
    for ax, (col, leak, title, cmap) in zip(axes[0], specs, strict=True):
        grid, xs, ys, mask = pivot_plane(df, col, "spectral_radius", "input_scaling", {"leak_rate": leak})
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        pcm = heatmap_plane(ax, grid, xs, ys, cmap, vmin=vmin, vmax=vmax, mask=mask)
        categorical_vline(ax, list(xs), 1.0, ls=":", color="#333", lw=1.0)
        ax.set_title(rf"{title}, $\alpha={leak:g}$")
        ax.set_xlabel(LAB_RHO)
        ax.set_ylabel(LAB_SIN)
        fmt = "{:.0f}" if col == "mc_total_mean" else "{:.2f}"
        _annotate_heatmap(ax, heatmap_display_grid(grid), vmin=vmin, vmax=vmax, cmap_name=cmap, fmt=fmt)
        add_colorbar(fig, ax, pcm, title)
    fig.subplots_adjust(wspace=0.55)
    fig.suptitle(r"Barrido multitarea en el plano $(\rho, s_{\mathrm{in}})$", y=1.04)
    return save_figure(fig, out_dir, "F_sweep_task_heatmaps", formats=formats)


def plot_sweep_aggregate_heatmaps(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """Meseta de compromiso — rank agregado en (rho, s_in) por leak.

    Light colormap (low rank = light = good); the contiguous light band is the
    robustness plateau. Auto-contrast annotations.
    """
    import matplotlib.pyplot as plt

    leaks = sorted(float(v) for v in df["leak_rate"].unique())
    cmap = CMAPS["sequential_cool"]  # low (good) -> light
    grids = []
    for leak in leaks:
        grid, xs, ys, mask = pivot_plane(df, "aggregate_rank", "spectral_radius", "input_scaling", {"leak_rate": leak})
        grids.append((grid, xs, ys, mask, leak))
    vmin = float(np.nanmin([np.nanmin(g[0]) for g in grids]))
    vmax = float(np.nanmax([np.nanmax(g[0]) for g in grids]))

    n = len(leaks)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6), squeeze=False)
    last_pcm = None
    for ax, (grid, xs, ys, mask, leak) in zip(axes[0], grids, strict=True):
        last_pcm = heatmap_plane(ax, grid, xs, ys, cmap, vmin=vmin, vmax=vmax, mask=mask)
        categorical_vline(ax, list(xs), 1.0, ls=":", color="#333", lw=1.0)
        ax.set_title(rf"Rank agregado, $\alpha={leak:g}$")
        ax.set_xlabel(LAB_RHO)
        ax.set_ylabel(LAB_SIN)
        _annotate_heatmap(ax, heatmap_display_grid(grid), vmin=vmin, vmax=vmax, cmap_name=cmap, fmt="{:.0f}")
    if last_pcm is not None:
        add_colorbar(fig, axes[0, -1], last_pcm, "Rank agregado")
    fig.suptitle(r"Rank agregado en el plano $(\rho, s_{\mathrm{in}})$", y=1.03)
    return save_figure(fig, out_dir, "F_sweep_aggregate_heatmaps", formats=formats)


def plot_sweep_tradeoff(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    y_metric: str,
    name: str,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """S3 — memory <-> nonlinearity trade-off: MC (x) vs error (y).

    Single panel, one marker, single sequential gradient by s_in (no marker zoo),
    with the Pareto frontier (max MC for each error level) drawn and the three
    representative configs labelled. The empty bottom-right corner is the no-free-
    lunch result.
    """
    import matplotlib.pyplot as plt

    if y_metric.startswith("mg"):
        ylabel, title = "Mackey-Glass NMSE test", "Relación memoria-error para Mackey-Glass"
    elif y_metric.startswith("narma"):
        ylabel, title = "NARMA-10 NMSE test", "Relación memoria-error para NARMA-10"
    else:
        ylabel, title = y_metric, "Relación memoria-error"

    sub = df.dropna(subset=["mc_total_mean", y_metric]).copy()
    x = sub["mc_total_mean"].to_numpy(float)
    y = sub[y_metric].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    sc = ax.scatter(x, y, c=sub["input_scaling"].astype(float), cmap="cividis",
                    s=34, alpha=0.85, edgecolors="white", linewidths=0.3)

    # Pareto frontier: for each point, non-dominated = no other has both higher MC and lower error.
    order = np.argsort(-x)  # high MC first
    best_err = np.inf
    front = []
    for i in order:
        if y[i] < best_err:
            front.append(i)
            best_err = y[i]
    front = sorted(front, key=lambda i: x[i])
    ax.plot(x[front], y[front], "-", color="#444", lw=1.2, alpha=0.7, zorder=1, label="frontera de Pareto")

    aggregate_idx = sub["aggregate_rank"].idxmin() if "aggregate_rank" in sub else None
    if aggregate_idx is not None:
        r = sub.loc[aggregate_idx]
        ax.annotate("mejor agregado", xy=(r["mc_total_mean"], r[y_metric]),
                    xytext=(8, -14), textcoords="offset points", fontsize=8, ha="left",
                    arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#333"})

    ax.margins(x=0.12)

    ax.set_yscale("log")
    ax.set_xlabel("Memory capacity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    leg = ax.legend(frameon=True, loc="upper left", fontsize=8)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_alpha(0.95)
    frame.set_linewidth(0.8)
    ax.grid(True, alpha=0.2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(LAB_SIN)
    return save_figure(fig, out_dir, name, formats=formats)


# ---------------------------------------------------------------------------
# Design figures (unchanged — redone in a later pass)
# ---------------------------------------------------------------------------

def plot_design_rank_heatmap(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    import matplotlib.pyplot as plt

    families = _ordered_families(df)
    cols = ["narma10", "mackey_glass", "memory_capacity", "aggregate"]
    mat = np.full((len(families), len(cols)), np.nan)
    for i, family in enumerate(families):
        sub = df[df["design_name"] == family]
        for j, task in enumerate(cols):
            if task == "aggregate":
                mat[i, j] = float(sub["global_aggregate_rank"].min())
            else:
                mat[i, j] = float(sub[TASK_RANK_COLS[task]].min())

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu_r")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([TASK_LABELS[c] for c in cols], rotation=20, ha="right")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([FAMILY_LABELS.get(f, f) for f in families])
    ax.set_title("Rank global por familia de reservorio")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Rank global")
    return save_figure(fig, out_dir, "F_design_family_rank_heatmap", formats=formats)


def plot_design_mg_vs_transient(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    import matplotlib.pyplot as plt

    _require_columns(df, ["diag_transient_growth_max_mean", "mg_val_nmse_mean"], "design diagnostics")
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for i, family in enumerate(_ordered_families(df)):
        sub = df[df["design_name"] == family].copy()
        x = np.log10(np.maximum(sub["diag_transient_growth_max_mean"].astype(float), 1e-12))
        ax.scatter(x, sub["mg_val_nmse_mean"], s=30, alpha=0.8,
                   color=PALETTE[i % len(PALETTE)], label=FAMILY_LABELS.get(family, family))
    ax.set_xlabel(r"$\log_{10}\max_k \|W^k\|_2$")
    ax.set_yscale("log")
    ax.set_ylabel("Mackey-Glass NMSE de validacion")
    ax.set_title("Transitorios no normales y error Mackey-Glass")
    ax.legend(frameon=False, ncols=2, fontsize=7)
    return save_figure(fig, out_dir, "F_design_mg_vs_transient", formats=formats)


def plot_design_mc_vs_henrici(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    import matplotlib.pyplot as plt

    _require_columns(df, ["diag_henrici_departure_mean", "mc_total_mean"], "design diagnostics")
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for i, family in enumerate(_ordered_families(df)):
        sub = df[df["design_name"] == family].copy()
        x = np.log10(np.maximum(sub["diag_henrici_departure_mean"].astype(float), 1e-12))
        ax.scatter(x, sub["mc_total_mean"], s=30, alpha=0.8,
                   color=PALETTE[i % len(PALETTE)], label=FAMILY_LABELS.get(family, family))
    ax.set_xlabel(r"$\log_{10}(\mathrm{Henrici}+10^{-12})$")
    ax.set_ylabel("Memory capacity")
    ax.set_title("Normalidad estructural y memoria lineal")
    ax.legend(frameon=False, ncols=2, fontsize=7)
    return save_figure(fig, out_dir, "F_design_mc_vs_henrici", formats=formats)


def plot_design_delta_tradeoff(
    df: pd.DataFrame,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    import matplotlib.pyplot as plt

    needed = ["delta_vs_baseline_mg_val_nmse", "delta_vs_baseline_mc_total"]
    if any(c not in df.columns for c in needed):
        return []
    sub_all = df[df["design_name"] != "random_sparse_baseline"].copy().dropna(subset=needed)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for i, family in enumerate(_ordered_families(sub_all)):
        sub = sub_all[sub_all["design_name"] == family]
        ax.scatter(sub["delta_vs_baseline_mc_total"], sub["delta_vs_baseline_mg_val_nmse"],
                   s=30, alpha=0.8, color=PALETTE[i % len(PALETTE)], label=FAMILY_LABELS.get(family, family))
    ax.axhline(0.0, lw=0.8, color="black", alpha=0.5)
    ax.axvline(0.0, lw=0.8, color="black", alpha=0.5)
    ax.set_xlabel(r"$\Delta$ MC frente a random sparse")
    ax.set_yscale("symlog", linthresh=0.02)
    ax.set_ylabel(r"$\Delta$ NMSE MG frente a random sparse (symlog)")
    ax.set_title("Diferencias frente a random sparse: memoria y error MG")
    ax.legend(frameon=False, ncols=2, fontsize=7)
    return save_figure(fig, out_dir, "F_design_delta_mc_mg", formats=formats)


# ---------------------------------------------------------------------------
# End-to-end builder
# ---------------------------------------------------------------------------

def build_campaign_artifacts(
    sweep_dir: str | Path,
    design_dir: str | Path,
    out_dir: str | Path,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> dict[str, list[Path] | dict[str, Path]]:
    """Build all standard tables and figures for sweep + design campaign."""
    sweep_df = load_sweep_summary(sweep_dir)
    design_df = load_design_summary(design_dir)
    out = Path(out_dir)
    tables = save_campaign_tables(sweep_df, design_df, out / "tables")
    sweep_figs = build_sweep_figures(sweep_df, out / "figures" / "sweep", formats=formats)
    design_figs = build_design_figures(design_df, out / "figures" / "design", formats=formats)
    return {"tables": tables, "sweep_figures": sweep_figs, "design_figures": design_figs}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ordered_families(df: pd.DataFrame) -> list[str]:
    present = [f for f in FAMILY_ORDER if f in set(df["design_name"].astype(str))]
    extras = sorted(set(df["design_name"].astype(str)) - set(present))
    return present + extras


def _annotate_heatmap(
    ax,
    grid: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str | None = None,
    fmt: str = "{:.0f}",
) -> None:
    """Annotate heatmap cells with auto-contrast text (white on dark, black on light)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return
    vmin = float(np.nanmin(grid)) if vmin is None else vmin
    vmax = float(np.nanmax(grid)) if vmax is None else vmax
    cmap = plt.get_cmap(cmap_name) if cmap_name else None
    norm = Normalize(vmin=vmin, vmax=vmax)
    for iy in range(grid.shape[0]):
        for ix in range(grid.shape[1]):
            val = grid[iy, ix]
            if not np.isfinite(val):
                continue
            color = "black"
            if cmap is not None:
                r, g, b, _ = cmap(norm(val))
                color = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
            ax.text(ix, iy, fmt.format(val), ha="center", va="center", fontsize=6.5, color=color)
