"""
Bug condition exploration tests — BEFORE implementing any fix.

These tests are EXPECTED TO FAIL on the current (unfixed) code.
Failure confirms the bugs exist. They are written as property-based
exploration tests following the bugfix workflow.

Bugs covered:
  - Bug 1: transient_kmax ignored by SweepRunner
  - Bug 2: Non-finite floats written as Infinity in JSON
  - Bug 3: mg_*_None_* columns in multitask summary CSV

**CRITICAL**: Do NOT attempt to fix the code when these tests fail.
Failure is the expected outcome on unfixed code.
"""

import csv
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rc_lab.runners.sweep_runner import SweepRunner, SweepRunResult
from rc_lab.runners.multitask_sweep_runner import (
    MultiTaskConfigEntry,
    MultiTaskSweepSummary,
    RankingSpec,
)
from rc_lab.utils.io import save_sweep_run_result, save_multitask_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_sweep_config(tmp_path: Path, transient_kmax: int = 10) -> dict:
    """
    Builds a minimal sweep config with diagnostics.transient_kmax set to the
    given value. Uses a tiny grid (1 config point, 1 seed, N=10, narma10).
    """
    return {
        "sweep": {
            "name": "bug1_exploration",
            "output_dir": str(tmp_path / "bug1_exploration"),
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
        "diagnostics": {
            "transient_kmax": transient_kmax,
        },
    }


def _make_sweep_run_result_with_inf() -> SweepRunResult:
    """
    Creates a SweepRunResult with a non-finite float in reservoir_diagnostics.
    singular_condition_number = inf simulates a near-singular reservoir matrix.
    """
    return SweepRunResult(
        sweep_name="bug2_exploration",
        config_id="abc123def456",
        seed=42,
        config_point={"spectral_radius": 0.9, "input_scaling": 0.1},
        best_ridge=1e-4,
        val_curve={"0.0001": 0.15},
        val_metrics={"nmse": 0.15},
        test_metrics={"nmse": 0.16},
        timing={"total_s": 0.5},
        timestamp=datetime.now(timezone.utc).isoformat(),
        reservoir_diagnostics={
            "singular_condition_number": float("inf"),
            "spectral_radius": 0.9,
        },
    )


def _make_multitask_summary_mg_disabled(tmp_path: Path) -> MultiTaskSweepSummary:
    """
    Builds a minimal MultiTaskSweepSummary with enabled_tasks=["narma10",
    "memory_capacity"] and one MultiTaskConfigEntry where mg_primary_metric=None,
    mg_val_mean=None, etc. (Mackey-Glass disabled).
    """
    entry = MultiTaskConfigEntry(
        config_id="abc123def456",
        config_point={"spectral_radius": 0.9, "input_scaling": 0.1},
        n_seeds=1,
        narma10_primary_metric="nmse",
        narma10_val_mean=0.15,
        narma10_val_std=0.01,
        narma10_test_mean=0.16,
        narma10_test_std=0.01,
        mg_primary_metric=None,
        mg_val_mean=None,
        mg_val_std=None,
        mg_test_mean=None,
        mg_test_std=None,
        mc_total_mean=45.0,
        mc_total_std=2.0,
        rank_narma10=1,
        rank_mg=None,
        rank_mc=1,
        aggregate_rank=1.0,
    )
    return MultiTaskSweepSummary(
        sweep_name="bug3_exploration",
        n_configs=1,
        n_seeds=1,
        grid={"spectral_radius": [0.9], "input_scaling": [0.1]},
        ranking_config={
            "narma10": RankingSpec(metric="nmse", direction="min"),
            "memory_capacity": RankingSpec(metric="mc_total", direction="max"),
        },
        configs=[entry],
        shortlist=["abc123def456"],
        shortlist_top_n=1,
        timestamp=datetime.now(timezone.utc).isoformat(),
        enabled_tasks=["narma10", "memory_capacity"],
    )


# ---------------------------------------------------------------------------
# Bug 1 exploration — transient_kmax ignored by SweepRunner
#
# Validates: Requirements 1.1
#
# EXPECTED OUTCOME ON UNFIXED CODE: FAILS
# Counterexample: SweepRunner calls reservoir_diagnostics with transient_kmax=50
# (the hardcoded default) instead of the configured value of 10.
# ---------------------------------------------------------------------------

def test_sweep_runner_transient_kmax_ignored_on_unfixed(tmp_path):
    """
    **Bug 1 exploration** — Property 1: Bug Condition

    Validates: Requirements 1.1

    Patches rc_lab.reservoirs.diagnostics.reservoir_diagnostics to capture
    the transient_kmax kwarg it receives. Builds a minimal sweep config with
    diagnostics.transient_kmax=10. Instantiates SweepRunner and calls run().

    EXPECTED OUTCOME ON UNFIXED CODE: FAILS
    Counterexample: SweepRunner calls reservoir_diagnostics with
    transient_kmax=50 (hardcoded default) instead of 10.
    """
    captured_kmax: list[int] = []

    # Import the real function so we can wrap it
    from rc_lab.reservoirs import diagnostics as diag_module

    original_fn = diag_module.reservoir_diagnostics

    def capturing_reservoir_diagnostics(W, transient_kmax=50):
        captured_kmax.append(transient_kmax)
        return original_fn(W, transient_kmax=transient_kmax)

    sweep_config = _make_minimal_sweep_config(tmp_path, transient_kmax=10)

    # Patch the function where SweepRunner imports it from
    with patch(
        "rc_lab.runners.sweep_runner._reservoir_diagnostics",
        side_effect=capturing_reservoir_diagnostics,
    ):
        runner = SweepRunner(sweep_config)
        runner.run()

    assert len(captured_kmax) >= 1, (
        "reservoir_diagnostics was never called — check the patch target"
    )

    # On UNFIXED code this assertion FAILS: captured value is 50, not 10
    assert captured_kmax[0] == 10, (
        f"COUNTEREXAMPLE: SweepRunner called reservoir_diagnostics with "
        f"transient_kmax={captured_kmax[0]} instead of the configured value 10. "
        f"Bug confirmed: SweepRunner ignores diagnostics.transient_kmax."
    )


# ---------------------------------------------------------------------------
# Bug 2 exploration — Non-finite floats written as Infinity in JSON
#
# Validates: Requirements 1.3, 1.4
#
# EXPECTED OUTCOME ON UNFIXED CODE: FAILS
# Counterexample: JSON file contains literal `Infinity` token.
# ---------------------------------------------------------------------------

def test_save_sweep_run_result_writes_infinity_on_unfixed(tmp_path):
    """
    **Bug 2 exploration** — Property 3: Bug Condition

    Validates: Requirements 1.3, 1.4

    Creates a SweepRunResult with reservoir_diagnostics containing
    singular_condition_number=float("inf"). Calls save_sweep_run_result.
    Reads the written file as raw text and asserts it does NOT contain
    the substring "Infinity".

    EXPECTED OUTCOME ON UNFIXED CODE: FAILS
    Counterexample: JSON file contains literal `Infinity` token, which is
    not valid JSON per RFC 8259 and causes json.loads() to fail.
    """
    result = _make_sweep_run_result_with_inf()
    save_sweep_run_result(result, tmp_path)

    # Find the written file
    runs_dir = tmp_path / "runs"
    json_files = list(runs_dir.glob("*.json"))
    assert len(json_files) == 1, f"Expected 1 JSON file, found {len(json_files)}"

    raw_text = json_files[0].read_text(encoding="utf-8")

    # On UNFIXED code this assertion FAILS: file contains "Infinity"
    assert "Infinity" not in raw_text, (
        f"COUNTEREXAMPLE: JSON file contains literal `Infinity` token. "
        f"Bug confirmed: save_sweep_run_result writes non-finite floats as "
        f"invalid JSON tokens instead of null. "
        f"Snippet: {raw_text[:200]!r}"
    )

    # Also verify the file is parseable by strict json.loads
    # (this will also fail on unfixed code if Infinity is present)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"COUNTEREXAMPLE: json.loads() raised JSONDecodeError on the written file: {e}. "
            f"Bug confirmed: file contains invalid JSON tokens."
        )


# ---------------------------------------------------------------------------
# Bug 3 exploration — mg_*_None_* columns in multitask summary CSV
#
# Validates: Requirements 1.5, 1.6
#
# EXPECTED OUTCOME ON UNFIXED CODE: FAILS
# Counterexample: CSV contains columns mg_val_None_mean, mg_test_None_mean, etc.
# ---------------------------------------------------------------------------

def test_save_multitask_summary_mg_none_columns_on_unfixed(tmp_path):
    """
    **Bug 3 exploration** — Property 5: Bug Condition

    Validates: Requirements 1.5, 1.6

    Builds a minimal MultiTaskSweepSummary with enabled_tasks=["narma10",
    "memory_capacity"] and one MultiTaskConfigEntry where mg_primary_metric=None.
    Calls save_multitask_summary. Reads summary.csv headers and asserts no
    column name contains the substring "None".

    EXPECTED OUTCOME ON UNFIXED CODE: FAILS
    Counterexample: CSV contains columns mg_val_None_mean, mg_val_None_std,
    mg_test_None_mean, mg_test_None_std, rank_mg — because save_multitask_summary
    unconditionally builds column names from entry.mg_primary_metric (which is
    None when Mackey-Glass is disabled).
    """
    summary = _make_multitask_summary_mg_disabled(tmp_path)
    save_multitask_summary(summary, tmp_path)

    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists(), f"summary.csv not found at {csv_path}"

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    none_columns = [col for col in fieldnames if "None" in col]

    # On UNFIXED code this assertion FAILS: columns like mg_val_None_mean are present
    assert not none_columns, (
        f"COUNTEREXAMPLE: CSV contains columns with 'None' in their name: {none_columns}. "
        f"Bug confirmed: save_multitask_summary builds column names from "
        f"entry.mg_primary_metric=None when Mackey-Glass is disabled, producing "
        f"malformed headers like mg_val_None_mean, mg_test_None_mean, etc. "
        f"All CSV columns: {fieldnames}"
    )
