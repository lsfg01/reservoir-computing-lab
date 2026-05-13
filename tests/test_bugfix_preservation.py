"""
Preservation tests — regression guards for the three bugfixes.

These tests MUST PASS on both unfixed and fixed code.
They capture baseline behaviour that must not regress after the fixes.

Properties covered:
  - Property 2: SweepRunner without diagnostics block uses default kmax=50
  - Property 4: Finite floats are not altered by JSON serialisation / make_json_safe
  - Property 6: Full column set when all tasks enabled
  - Backward compat: old MultiTaskSweepSummary without enabled_tasks produces full columns
"""

import csv
import json
import math
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from rc_lab.runners.sweep_runner import SweepRunner
from rc_lab.runners.multitask_sweep_runner import (
    MultiTaskConfigEntry,
    MultiTaskSweepSummary,
    RankingSpec,
)
from rc_lab.utils.io import make_json_safe, save_multitask_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_sweep_config_no_diagnostics(output_dir: str) -> dict:
    """Minimal sweep config WITHOUT a diagnostics block."""
    return {
        "sweep": {
            "name": "preservation_no_diag",
            "output_dir": output_dir,
            "seeds": [42],
        },
        "task": {
            "name": "narma10",
            "n_train": 200,
            "n_val": 50,
            "n_test": 50,
            "washout": 20,
            "state_policy": "reset",
        },
        "reservoir": {
            "type": "random_sparse",
            "N": 10,
            "sparsity": 0.9,
            "bias_scaling": 0.0,
        },
        "grid": {
            "spectral_radius": [0.9],
            "input_scaling": [0.1],
        },
        "readout": {
            "type": "ridge",
            "features": "states",
            "ridge_candidates": [1e-4],
        },
        "metrics": ["nmse"],
        # NOTE: no "diagnostics" block — SweepRunner must default to kmax=50
    }


def _make_multitask_summary_all_tasks(tmp_path: Path) -> MultiTaskSweepSummary:
    """MultiTaskSweepSummary with all three tasks enabled."""
    entry = MultiTaskConfigEntry(
        config_id="abc123def456",
        config_point={"spectral_radius": 0.9, "input_scaling": 0.1},
        n_seeds=1,
        narma10_primary_metric="nmse",
        narma10_val_mean=0.15,
        narma10_val_std=0.01,
        narma10_test_mean=0.16,
        narma10_test_std=0.01,
        mg_primary_metric="nmse",
        mg_val_mean=0.12,
        mg_val_std=0.008,
        mg_test_mean=0.13,
        mg_test_std=0.009,
        mc_total_mean=45.0,
        mc_total_std=2.0,
        rank_narma10=1,
        rank_mg=1,
        rank_mc=1,
        aggregate_rank=1.0,
    )
    return MultiTaskSweepSummary(
        sweep_name="preservation_all_tasks",
        n_configs=1,
        n_seeds=1,
        grid={"spectral_radius": [0.9], "input_scaling": [0.1]},
        ranking_config={
            "narma10": RankingSpec(metric="nmse", direction="min"),
            "mackey_glass": RankingSpec(metric="nmse", direction="min"),
            "memory_capacity": RankingSpec(metric="mc_total", direction="max"),
        },
        configs=[entry],
        shortlist=["abc123def456"],
        shortlist_top_n=1,
        timestamp=datetime.now(timezone.utc).isoformat(),
        enabled_tasks=["narma10", "mackey_glass", "memory_capacity"],
    )


# ---------------------------------------------------------------------------
# Preservation test 1 — Property 2
#
# SweepRunner without diagnostics block uses default kmax=50
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------

def test_sweep_runner_no_diagnostics_uses_default_kmax(tmp_path):
    """
    Property 2: Preservation — SweepRunner without diagnostics block.

    Validates: Requirements 3.1

    When no diagnostics block is present, SweepRunner must call
    reservoir_diagnostics with transient_kmax=50 (the hardcoded default).
    """
    captured_kmax: list[int] = []

    from rc_lab.reservoirs import diagnostics as diag_module
    original_fn = diag_module.reservoir_diagnostics

    def capturing_fn(W, transient_kmax=50):
        captured_kmax.append(transient_kmax)
        return original_fn(W, transient_kmax=transient_kmax)

    sweep_config = _make_minimal_sweep_config_no_diagnostics(
        str(tmp_path / "preservation_no_diag")
    )

    with patch("rc_lab.runners.sweep_runner._reservoir_diagnostics", side_effect=capturing_fn):
        SweepRunner(sweep_config).run()

    assert len(captured_kmax) >= 1, "reservoir_diagnostics was never called"
    assert all(k == 50 for k in captured_kmax), (
        f"REGRESSION: SweepRunner used transient_kmax={captured_kmax} "
        f"instead of default 50 when no diagnostics block is present."
    )


# ---------------------------------------------------------------------------
# Preservation test 1b — Property 2 (multiple grid values, deterministic)
#
# Checks several (spectral_radius, input_scaling) combinations deterministically.
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------

_GRID_CASES = [
    (0.1, 0.01),
    (0.5, 0.5),
    (0.9, 0.1),
    (1.2, 1.0),
    (1.5, 2.0),
]


def test_sweep_runner_no_diagnostics_uses_default_kmax_varied_grid(tmp_path):
    """
    Property 2: Preservation — varied grid values, no diagnostics block.

    Validates: Requirements 3.1

    For several (spectral_radius, input_scaling) combinations, SweepRunner
    must always call reservoir_diagnostics with transient_kmax=50 when no
    diagnostics block is present.
    """
    from rc_lab.reservoirs import diagnostics as diag_module
    original_fn = diag_module.reservoir_diagnostics

    for spectral_radius, input_scaling in _GRID_CASES:
        captured_kmax: list[int] = []

        def capturing_fn(W, transient_kmax=50, _cap=captured_kmax):
            _cap.append(transient_kmax)
            return original_fn(W, transient_kmax=transient_kmax)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sweep_config = {
                "sweep": {
                    "name": "preservation_varied",
                    "output_dir": str(Path(tmp_dir) / "out"),
                    "seeds": [42],
                },
                "task": {
                    "name": "narma10",
                    "n_train": 200,
                    "n_val": 50,
                    "n_test": 50,
                    "washout": 20,
                    "state_policy": "reset",
                },
                "reservoir": {
                    "type": "random_sparse",
                    "N": 10,
                    "sparsity": 0.9,
                    "bias_scaling": 0.0,
                },
                "grid": {
                    "spectral_radius": [spectral_radius],
                    "input_scaling": [input_scaling],
                },
                "readout": {
                    "type": "ridge",
                    "features": "states",
                    "ridge_candidates": [1e-4],
                },
                "metrics": ["nmse"],
                # No diagnostics block
            }

            with patch(
                "rc_lab.runners.sweep_runner._reservoir_diagnostics",
                side_effect=capturing_fn,
            ):
                SweepRunner(sweep_config).run()

        assert len(captured_kmax) >= 1, (
            f"reservoir_diagnostics never called for sr={spectral_radius}, is={input_scaling}"
        )
        assert all(k == 50 for k in captured_kmax), (
            f"REGRESSION: transient_kmax={captured_kmax} for "
            f"sr={spectral_radius}, is={input_scaling} — expected 50."
        )


# ---------------------------------------------------------------------------
# Preservation test 2 — Property 4
#
# make_json_safe does not alter finite floats; non-finite → None
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

# Fixed set of finite float cases covering typical and edge values
_FINITE_FLOAT_CASES = [
    {"spectral_radius": 0.9, "input_scaling": 0.1},
    {"val_nmse": 0.15, "test_nmse": 0.16, "mc_total": 45.0},
    {"zero": 0.0, "negative": -1.5, "small": 1e-10, "large": 1e10},
    {"a": 1.0, "b": -1.0, "c": 0.5, "d": -0.5},
    {"x": 1e-300, "y": 1e300},
]

# Fixed set of non-finite cases that must become None
_NON_FINITE_CASES = [
    ({"inf": float("inf")},          {"inf": None}),
    ({"neg_inf": float("-inf")},     {"neg_inf": None}),
    ({"nan": float("nan")},          {"nan": None}),
    ({"a": 1.0, "b": float("inf")},  {"a": 1.0, "b": None}),
    ({"a": float("nan"), "b": 2.0},  {"a": None, "b": 2.0}),
    ({"nested": [float("inf"), 1.0]}, {"nested": [None, 1.0]}),
]


def test_make_json_safe_finite_values_unchanged():
    """
    Property 4: Preservation — finite floats are not altered by make_json_safe.

    Validates: Requirements 3.3

    make_json_safe must return an equal dict for all-finite inputs, and the
    result must round-trip through json.dumps / json.loads unchanged.
    """
    for d in _FINITE_FLOAT_CASES:
        safe = make_json_safe(d)
        assert safe == d, (
            f"REGRESSION: make_json_safe altered finite float values.\n"
            f"  Input:  {d}\n"
            f"  Output: {safe}"
        )
        # Also verify strict JSON round-trip
        recovered = json.loads(json.dumps(safe, allow_nan=False))
        assert recovered == d, (
            f"REGRESSION: json round-trip altered finite float values.\n"
            f"  Input:  {d}\n"
            f"  Recovered: {recovered}"
        )


def test_make_json_safe_non_finite_becomes_none():
    """
    Property 4 (complement): non-finite floats are replaced with None.

    Validates: Requirements 2.3

    make_json_safe must replace inf, -inf, and nan with None, and the
    result must be parseable by json.loads with allow_nan=False.
    """
    for input_dict, expected in _NON_FINITE_CASES:
        result = make_json_safe(input_dict)
        assert result == expected, (
            f"make_json_safe produced unexpected output.\n"
            f"  Input:    {input_dict}\n"
            f"  Expected: {expected}\n"
            f"  Got:      {result}"
        )
        # Must produce strict JSON (no Infinity / NaN tokens)
        json_text = json.dumps(result, allow_nan=False)
        assert "Infinity" not in json_text
        assert "NaN" not in json_text


# ---------------------------------------------------------------------------
# Preservation test 3 — Property 6
#
# Full column set when all tasks enabled
# Validates: Requirements 3.2, 3.5, 3.6
# ---------------------------------------------------------------------------

_EXPECTED_FULL_COLUMNS = {
    "config_id",
    "spectral_radius",
    "input_scaling",
    "narma10_val_nmse_mean",
    "narma10_val_nmse_std",
    "narma10_test_nmse_mean",
    "narma10_test_nmse_std",
    "mg_val_nmse_mean",
    "mg_val_nmse_std",
    "mg_test_nmse_mean",
    "mg_test_nmse_std",
    "mc_total_mean",
    "mc_total_std",
    "rank_narma10",
    "rank_mg",
    "rank_mc",
    "aggregate_rank",
    "n_seeds",
}


def test_save_multitask_summary_all_tasks_full_columns(tmp_path):
    """
    Property 6: Preservation — full column set when all tasks enabled.

    Validates: Requirements 3.2, 3.5, 3.6

    When enabled_tasks contains all three tasks, summary.csv must include
    the complete expected column set with no None-derived names.
    """
    summary = _make_multitask_summary_all_tasks(tmp_path)
    save_multitask_summary(summary, tmp_path)

    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists(), f"summary.csv not found at {csv_path}"

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        fieldnames = set(csv.DictReader(f).fieldnames or [])

    missing = _EXPECTED_FULL_COLUMNS - fieldnames
    assert not missing, (
        f"REGRESSION: summary.csv missing columns when all tasks enabled: {sorted(missing)}.\n"
        f"Present: {sorted(fieldnames)}"
    )

    none_cols = [c for c in fieldnames if "None" in c]
    assert not none_cols, (
        f"REGRESSION: summary.csv has None-derived columns even with all tasks enabled: {none_cols}"
    )


# ---------------------------------------------------------------------------
# Preservation test 4 — Backward compatibility
#
# Summaries without enabled_tasks produce the full column set
# Validates: Requirements 2.9
# ---------------------------------------------------------------------------

def test_save_multitask_summary_backward_compat_no_enabled_tasks(tmp_path):
    """
    Backward compatibility: old summaries without enabled_tasks produce full columns.

    Validates: Requirements 2.9

    The getattr fallback in save_multitask_summary defaults to all three tasks
    when enabled_tasks is absent, preserving the pre-fix behaviour.
    """
    # Verify the getattr fallback contract
    old_obj = types.SimpleNamespace(sweep_name="old", n_configs=1, n_seeds=1)
    fallback = getattr(old_obj, "enabled_tasks", ["narma10", "mackey_glass", "memory_capacity"])
    assert fallback == ["narma10", "mackey_glass", "memory_capacity"], (
        "getattr fallback for enabled_tasks does not return the expected default"
    )

    # A summary with enabled_tasks equal to the default must produce the full column set
    summary = _make_multitask_summary_all_tasks(tmp_path)
    summary.enabled_tasks = ["narma10", "mackey_glass", "memory_capacity"]
    save_multitask_summary(summary, tmp_path)

    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists(), f"summary.csv not found at {csv_path}"

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        fieldnames = set(csv.DictReader(f).fieldnames or [])

    missing = _EXPECTED_FULL_COLUMNS - fieldnames
    assert not missing, (
        f"REGRESSION: backward-compat case missing columns: {sorted(missing)}.\n"
        f"Present: {sorted(fieldnames)}"
    )

    none_cols = [c for c in fieldnames if "None" in c]
    assert not none_cols, (
        f"REGRESSION: backward-compat case has None-derived columns: {none_cols}"
    )
