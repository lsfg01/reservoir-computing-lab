"""
style.py — Global style, palette and figure-saving utilities.

matplotlib.use("Agg") is called at import time so this module is safe to use
in headless environments and test suites.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

PALETTE: list[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]

CMAPS: dict[str, str] = {
    "sequential_warm": "YlOrRd",
    "sequential_cool": "YlGnBu",
    "diverging": "RdBu_r",
}


def apply_style() -> None:
    """Apply rc-lab house style to all subsequent matplotlib figures."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })


def save_figure(
    fig: "plt.Figure",
    out_dir: str | Path,
    name: str,
    formats: Sequence[str] = ("svg", "pdf", "png"),
    dpi: int = 150,
) -> list[Path]:
    """
    Save fig to out_dir/name.{fmt} for each fmt in formats.

    Closes the figure after saving. Returns list of written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
