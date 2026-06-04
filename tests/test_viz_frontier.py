"""
Tests for the viz.frontier pipeline and the CSV output of ESPFrontierRunner.

All tests use Agg backend (imported via rc_lab.viz.style) and synthetic data;
they do NOT run the full frontier sweep.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RHO_VALUES = [0.5, 0.9, 1.0, 1.3, 1.5]
_SIN_VALUES = [0.1, 0.4, 0.8]
_ALPHA_VALUES = [0.5, 1.0]
_FAMILY = "random_sparse"


def _make_frac_sync(rho: float, s_in: float) -> float:
    """Synthetic sync: drops sharply above rho=1."""
    if rho <= 0.95:
        return 1.0
    if rho <= 1.05:
        return 0.6
    return 0.0


def _make_sync_time(frac: float) -> float | None:
    if frac >= 0.999:
        return 100.0 + np.random.default_rng(42).uniform(0, 50)
    return None


def _synthetic_summary_csv(tmp_path: Path) -> Path:
    rows = []
    for alpha in _ALPHA_VALUES:
        for s_in in _SIN_VALUES:
            for rho in _RHO_VALUES:
                frac = _make_frac_sync(rho, s_in)
                st = 120.0 if frac >= 0.999 else None
                nfd = 0.7 if (0.05 < rho - 1.0 <= 0.3 and frac < 0.999) else (0.0 if frac >= 0.999 else None)
                rows.append({
                    "family_name": _FAMILY,
                    "s_in": s_in,
                    "alpha": alpha,
                    "rho_target": rho,
                    "n_seeds": 7,
                    "fraction_synchronized_mean": frac,
                    "fraction_synchronized_std": 0.0,
                    "sync_time_mean_mean": st,
                    "sync_time_mean_std": 5.0 if st is not None else None,
                    "sigma_max_mean": rho * 1.05,
                    "rho_real_mean": rho,
                    "saturation_mean_mean": min(0.99, s_in * rho),
                    "saturation_frac_mean": 0.1 * s_in,
                    "nonsync_fraction_descending": nfd,
                })
    csv_path = tmp_path / "summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def _synthetic_summary_json(tmp_path: Path) -> Path:
    """Minimal summary.json (same rows) for bootstrap test."""
    rows = []
    for alpha in [1.0]:
        for s_in in [0.1]:
            for rho in [0.5, 1.5]:
                frac = 1.0 if rho < 1.0 else 0.0
                rows.append({
                    "family_name": _FAMILY,
                    "s_in": s_in,
                    "alpha": alpha,
                    "rho_target": rho,
                    "n_seeds": 2,
                    "fraction_synchronized_mean": frac,
                    "fraction_synchronized_std": 0.0,
                    "sync_time_mean_mean": 80.0 if frac >= 0.999 else None,
                    "sync_time_mean_std": None,
                    "sigma_max_mean": rho,
                    "rho_real_mean": rho,
                    "saturation_mean_mean": 0.3,
                    "saturation_frac_mean": 0.05,
                    "nonsync_fraction_descending": None,
                })
    json_path = tmp_path / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"name": "test", "rows": rows}, f)
    return json_path


# ---------------------------------------------------------------------------
# Tests: io.load_summary
# ---------------------------------------------------------------------------

def test_load_summary_reads_csv(tmp_path: Path) -> None:
    csv_path = _synthetic_summary_csv(tmp_path)
    from rc_lab.viz.io import load_summary

    df = load_summary(csv_path)
    expected_rows = len(_ALPHA_VALUES) * len(_SIN_VALUES) * len(_RHO_VALUES)
    assert len(df) == expected_rows
    assert "fraction_synchronized_mean" in df.columns


# ---------------------------------------------------------------------------
# Tests: io.pivot_plane
# ---------------------------------------------------------------------------

def test_pivot_plane_shape_and_mask(tmp_path: Path) -> None:
    csv_path = _synthetic_summary_csv(tmp_path)
    from rc_lab.viz.io import load_summary, pivot_plane

    df = load_summary(csv_path)
    grid, xs, ys, mask = pivot_plane(
        df, "fraction_synchronized_mean", "rho_target", "s_in",
        {"alpha": 1.0, "family_name": _FAMILY},
    )

    assert grid.shape == (len(_SIN_VALUES), len(_RHO_VALUES))
    assert len(xs) == len(_RHO_VALUES)
    assert len(ys) == len(_SIN_VALUES)
    # x ascending
    assert xs == sorted(xs)
    # y descending
    assert ys == sorted(ys, reverse=True)
    # No NaN in this fixture
    assert not np.any(mask)


def test_pivot_plane_nan_mask(tmp_path: Path) -> None:
    """A row with a NaN value_col should produce True in mask."""
    df = pd.DataFrame([
        {"family_name": _FAMILY, "alpha": 1.0, "rho_target": 0.5, "s_in": 0.1, "val": 0.8},
        {"family_name": _FAMILY, "alpha": 1.0, "rho_target": 1.5, "s_in": 0.1, "val": float("nan")},
    ])
    from rc_lab.viz.io import pivot_plane

    grid, xs, ys, mask = pivot_plane(df, "val", "rho_target", "s_in",
                                     {"alpha": 1.0, "family_name": _FAMILY})
    assert mask.shape == grid.shape
    # The NaN cell should be masked
    assert np.any(mask)


# ---------------------------------------------------------------------------
# Tests: io.frontier_per_x
# ---------------------------------------------------------------------------

def test_frontier_per_x_finds_rho_star(tmp_path: Path) -> None:
    """For s_in=0.1 @ alpha=1.0, expected rho* = 0.9 (last fully-sync rho)."""
    csv_path = _synthetic_summary_csv(tmp_path)
    from rc_lab.viz.io import frontier_per_x, load_summary

    df = load_summary(csv_path)
    fp = frontier_per_x(
        df, "rho_target", "s_in", "fraction_synchronized_mean", 0.999,
        {"alpha": 1.0, "family_name": _FAMILY},
    )
    # rho=0.5 and 0.9 both have frac=1.0; so max s_in for rho in {0.5,0.9} = 0.8
    assert 0.5 in fp
    assert 0.9 in fp
    # rho=1.0 has frac=0.6 → not included
    assert 1.0 not in fp

    # max s_in for rho=0.5 should be 0.8 (all s_in sync at rho=0.5)
    assert fp[0.5] == pytest.approx(max(_SIN_VALUES))


def test_frontier_per_x_empty_when_nothing_valid(tmp_path: Path) -> None:
    df = pd.DataFrame([
        {"family_name": _FAMILY, "alpha": 1.0, "rho_target": 1.5, "s_in": 0.1,
         "fraction_synchronized_mean": 0.0}
    ])
    from rc_lab.viz.io import frontier_per_x

    fp = frontier_per_x(df, "rho_target", "s_in", "fraction_synchronized_mean",
                        0.999, {"alpha": 1.0, "family_name": _FAMILY})
    assert fp == {}


def test_frontier_per_sin_increases_on_synthetic_grid() -> None:
    """rho*(s_in) must reduce over rho, not over s_in."""
    rhos = [0.5, 0.8, 1.0, 1.2]
    rho_star_by_sin = {0.1: 0.5, 0.4: 0.8, 0.8: 1.2}
    rows = []
    for s_in, rho_star in rho_star_by_sin.items():
        for rho in rhos:
            rows.append({
                "family_name": _FAMILY,
                "alpha": 1.0,
                "rho_target": rho,
                "s_in": s_in,
                "fraction_synchronized_mean": 1.0 if rho <= rho_star else 0.0,
            })
    df = pd.DataFrame(rows)

    from rc_lab.viz.io import frontier_per_sin

    fp = frontier_per_sin(
        df, "rho_target", "s_in", "fraction_synchronized_mean",
        0.999, {"alpha": 1.0, "family_name": _FAMILY},
    )
    frontier = [fp[s_in] for s_in in sorted(rho_star_by_sin)]

    assert frontier == pytest.approx([0.5, 0.8, 1.2])
    assert frontier == sorted(frontier)
    assert len(set(frontier)) > 1


# ---------------------------------------------------------------------------
# Tests: bootstrap json→csv
# ---------------------------------------------------------------------------

def test_bootstrap_json_to_csv(tmp_path: Path) -> None:
    """scripts/plot_frontier._bootstrap_csv regenerates CSV from sibling JSON."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "plot_frontier",
        Path(__file__).parent.parent / "scripts" / "plot_frontier.py",
    )
    mod = importlib.util.load_from_spec = None  # suppress unused
    pf = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["plot_frontier"] = pf
    spec.loader.exec_module(pf)  # type: ignore[union-attr]

    json_path = _synthetic_summary_json(tmp_path)
    csv_path = tmp_path / "summary.csv"
    assert not csv_path.exists()

    result = pf._bootstrap_csv(csv_path)
    assert result is True
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert "fraction_synchronized_mean" in df.columns
    assert len(df) == 2  # 1 alpha × 1 s_in × 2 rho


# ---------------------------------------------------------------------------
# Tests: figure builders write 3 formats without error
# ---------------------------------------------------------------------------

@pytest.fixture()
def summary_csv(tmp_path: Path) -> Path:
    return _synthetic_summary_csv(tmp_path)


def _check_formats(paths: list[Path], name_prefix: str) -> None:
    exts = {p.suffix for p in paths}
    assert ".svg" in exts, f"Missing .svg for {name_prefix}"
    assert ".pdf" in exts, f"Missing .pdf for {name_prefix}"
    assert ".png" in exts, f"Missing .png for {name_prefix}"
    for p in paths:
        assert p.exists(), f"File not written: {p}"
        assert p.stat().st_size > 0, f"Empty file: {p}"


def test_f1_writes_three_formats(summary_csv: Path, tmp_path: Path) -> None:
    from rc_lab.viz.frontier import _f1_frac_sync
    from rc_lab.viz.io import load_summary

    df = load_summary(summary_csv)
    paths = _f1_frac_sync(df, tmp_path / "figs", _FAMILY, 1.0)
    _check_formats(paths, "F1")


def test_f2_writes_three_formats(summary_csv: Path, tmp_path: Path) -> None:
    from rc_lab.viz.frontier import CANDIDATE_REGIONS, _f2_washout_heatmap
    from rc_lab.viz.io import load_summary

    df = load_summary(summary_csv)
    paths = _f2_washout_heatmap(df, tmp_path / "figs", _FAMILY, 1.0, CANDIDATE_REGIONS)
    _check_formats(paths, "F2")


def test_f3_writes_three_formats(summary_csv: Path, tmp_path: Path) -> None:
    from rc_lab.viz.frontier import CANDIDATE_REGIONS, _f3_sat_heatmap
    from rc_lab.viz.io import load_summary

    df = load_summary(summary_csv)
    paths = _f3_sat_heatmap(df, tmp_path / "figs", _FAMILY, 1.0, CANDIDATE_REGIONS)
    _check_formats(paths, "F3")


def test_f4_writes_three_formats(summary_csv: Path, tmp_path: Path) -> None:
    from rc_lab.viz.frontier import _f4_rho_star_curves
    from rc_lab.viz.io import load_summary

    df = load_summary(summary_csv)
    paths = _f4_rho_star_curves(df, tmp_path / "figs", _FAMILY)
    _check_formats(paths, "F4")


def test_f5_writes_three_formats(summary_csv: Path, tmp_path: Path) -> None:
    from rc_lab.viz.frontier import _f5_washout_lines
    from rc_lab.viz.io import load_summary

    df = load_summary(summary_csv)
    paths = _f5_washout_lines(df, tmp_path / "figs", _FAMILY, 1.0)
    _check_formats(paths, "F5")


def test_f6_writes_three_formats(summary_csv: Path, tmp_path: Path) -> None:
    from rc_lab.viz.frontier import _f6_leak_cuts
    from rc_lab.viz.io import load_summary

    df = load_summary(summary_csv)
    paths = _f6_leak_cuts(df, tmp_path / "figs", _FAMILY)
    _check_formats(paths, "F6")


def test_f7_writes_three_formats(summary_csv: Path, tmp_path: Path) -> None:
    from rc_lab.viz.frontier import _f7_surface_3d
    from rc_lab.viz.io import load_summary

    df = load_summary(summary_csv)
    paths = _f7_surface_3d(df, tmp_path / "figs", _FAMILY)
    _check_formats(paths, "F7")


def test_build_frontier_figures_returns_paths(summary_csv: Path, tmp_path: Path) -> None:
    """build_frontier_figures (F1-F7, no runs_dir) returns >= 7*3 paths."""
    from rc_lab.viz.frontier import build_frontier_figures

    paths = build_frontier_figures(summary_csv, tmp_path / "figs")
    assert len(paths) >= 7 * 3
    for p in paths:
        assert p.exists()


# ---------------------------------------------------------------------------
# Tests: runner CSV output
# ---------------------------------------------------------------------------

def test_runner_run_writes_summary_csv(tmp_path: Path) -> None:
    """After ESPFrontierRunner.run(), summary.csv must exist with expected columns."""
    from rc_lab.runners.esp_frontier_runner import ESPFrontierRunner

    cfg = {
        "frontier": {
            "name": "csv_test",
            "output_dir": str(tmp_path),
            "seeds": [42],
        },
        "esp": {"T": 100, "n_pairs": 2, "eps": 1e-3},
        "families": [
            {"name": "random_sparse", "type": "random_sparse", "N": 20,
             "sparsity": 0.9, "bias_scaling": 0.0}
        ],
        "grid": {
            "spectral_radius": [0.5, 0.9],
            "input_scaling": [0.1],
            "leak_rate": [1.0],
        },
    }
    ESPFrontierRunner(cfg).run()

    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists(), "summary.csv not written"

    df = pd.read_csv(csv_path)
    expected_cols = {
        "family_name", "s_in", "alpha", "rho_target", "n_seeds",
        "fraction_synchronized_mean", "fraction_synchronized_std",
        "sync_time_mean_mean", "sigma_max_mean", "rho_real_mean",
        "saturation_mean_mean", "saturation_frac_mean",
        "nonsync_fraction_descending",
    }
    assert expected_cols.issubset(set(df.columns)), (
        f"Missing columns: {expected_cols - set(df.columns)}"
    )
    # One row per (family × s_in × alpha × rho)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Tests: amplitude panel does NOT contain the word "saturación"
# ---------------------------------------------------------------------------

def test_f3_amplitude_panel_not_saturacion(summary_csv: Path, tmp_path: Path) -> None:
    """F3 colorbar and title must use 'amplitud', not 'saturación'."""
    import unittest.mock

    import matplotlib.pyplot as plt
    import rc_lab.viz.frontier as frontier_mod
    from rc_lab.viz.frontier import CANDIDATE_REGIONS, _f3_sat_heatmap
    from rc_lab.viz.io import load_summary

    captured: list[plt.Figure] = []

    def _capture(fig: plt.Figure, *args: Any, **kwargs: Any) -> list:
        captured.append(fig)
        return []

    with unittest.mock.patch.object(frontier_mod, "save_figure", side_effect=_capture):
        df = load_summary(summary_csv)
        _f3_sat_heatmap(df, tmp_path / "figs", _FAMILY, 1.0, CANDIDATE_REGIONS)

    assert captured, "No figure was captured"
    fig = captured[0]

    all_text: list[str] = []
    for ax in fig.axes:
        all_text.append(ax.title.get_text())
        all_text.append(ax.xaxis.label.get_text())
        all_text.append(ax.yaxis.label.get_text())
        for t in ax.texts:
            all_text.append(t.get_text())

    combined = " ".join(all_text).lower()
    assert "saturación" not in combined, (
        f"Found 'saturación' in amplitude panel text: {combined!r}"
    )
    assert "amplitud" in combined or "langle" in combined.replace(" ", ""), (
        f"Expected 'amplitud' or LaTeX angle bracket in text: {combined!r}"
    )
    plt.close(fig)
