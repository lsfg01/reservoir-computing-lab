"""
Preservation property tests — BEFORE implementing any fix.

These tests MUST PASS on the current (unfixed) code.
They capture baseline behaviour that must not regress after the fixes are applied.

Properties covered:
  - Property 2: SweepRunner without diagnostics block uses default kmax=50
  - Property 4: Finite floats are not altered by JSON serialisation
  - Property 6: Full column set when all tasks enabled
  - Backward compat: old MultiTaskSweepSummary without enabled_tasks produces full columns

**IMPORTANT**: All tests here must PASS on unfixed code.
They are regression guards — if any of these fail after a fix, the fix introduced a regression.
"""

import csv
import dataclasses
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rc_lab.runners.sweep_runner import SweepRunner
from rc_lab.runners.multitask_sweep_runner import (
    MultiTaskConfigEntry,
    MultiTaskSweepSummary,
    RankingSpec,
)
from rc_lab.utils.io import save_multitask_summary


# ---------------------------------------------------------------------------
# Helpers (shared with exploration tests)
# ---------------------------------------------------------------------------

def _make_minimal_sweep_config_no_diagnostics(tmp_path: Path) -> dict:
    """
    Builds a minimal sweep config WITHOUT a diagnostics block.
    SweepRunner should use the default transient_kmax=50 in this case.
    """
    return {
        "sweep": {
            "name": "preservation_no_diag",
            "output_dir": str(tmp_path / "preservation_no_diag"),
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
    """
    Builds a MultiTaskSweepSummary with all three tasks enabled and
    mg_primary_metric="nmse". This is the non-bug-condition case for Bug 3.
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
#
# Validates: Requirements 3.1
#
# EXPECTED OUTCOME: PASSES on both unfixed and fixed code.
# ---------------------------------------------------------------------------

def test_sweep_runner_no_diagnostics_uses_default_kmax(tmp_path):
    """
    **Preservation test 1** — Property 2: Preservation

    Validates: Requirements 3.1

    For any sweep config without a diagnostics block, SweepRunner must call
    reservoir_diagnostics with transient_kmax=50 (the default).

    This test MUST PASS on unfixed code (baseline behaviour) and MUST CONTINUE
    TO PASS after the Bug 1 fix is applied (no regression).
    """
    captured_kmax: list[int] = []

    from rc_lab.reservoirs import diagnostics as diag_module

    original_fn = diag_module.reservoir_diagnostics

    def capturing_reservoir_diagnostics(W, transient_kmax=50):
        captured_kmax.append(transient_kmax)
        return original_fn(W, transient_kmax=transient_kmax)

    sweep_config = _make_minimal_sweep_config_no_diagnostics(tmp_path)

    with patch(
        "rc_lab.runners.sweep_runner._reservoir_diagnostics",
        side_effect=capturing_reservoir_diagnostics,
    ):
        runner = SweepRunner(sweep_config)
        runner.run()

    assert len(captured_kmax) >= 1, (
        "reservoir_diagnostics was never called — check the patch target"
    )

    # Preservation: no diagnostics block → must use default kmax=50
    assert captured_kmax[0] == 50, (
        f"REGRESSION: SweepRunner called reservoir_diagnostics with "
        f"transient_kmax={captured_kmax[0]} instead of the default 50 "
        f"when no diagnostics block is present in the config."
    )


# ---------------------------------------------------------------------------
# Preservation test 1 — Property 2 (property-based variant)
#
# For any sweep config without a diagnostics block, the captured transient_kmax
# equals 50. Uses hypothesis to vary the grid values.
#
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------

@given(
    spectral_radius=st.floats(min_value=0.1, max_value=1.5, allow_nan=False, allow_infinity=False),
    input_scaling=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=10, deadline=60_000)
def test_sweep_runner_no_diagnostics_uses_default_kmax_property(
    spectral_radius,
    input_scaling,
):
    """
    **Property 2: Preservation** (property-based)

    Validates: Requirements 3.1

    For any sweep config without a diagnostics block (varying spectral_radius
    and input_scaling), SweepRunner must call reservoir_diagnostics with
    transient_kmax=50.

    This property MUST HOLD on both unfixed and fixed code.
    """
    captured_kmax: list[int] = []

    from rc_lab.reservoirs import diagnostics as diag_module

    original_fn = diag_module.reservoir_diagnostics

    def capturing_reservoir_diagnostics(W, transient_kmax=50):
        captured_kmax.append(transient_kmax)
        return original_fn(W, transient_kmax=transient_kmax)

    # Use a temporary directory managed inline (not via fixture) so hypothesis
    # can reset state between examples without fixture scoping issues.
    with tempfile.TemporaryDirectory() as tmp_dir:
        sweep_config = {
            "sweep": {
                "name": "preservation_pbt",
                "output_dir": str(Path(tmp_dir) / "preservation_pbt"),
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
            side_effect=capturing_reservoir_diagnostics,
        ):
            runner = SweepRunner(sweep_config)
            runner.run()

    assert len(captured_kmax) >= 1
    assert all(k == 50 for k in captured_kmax), (
        f"REGRESSION: SweepRunner used transient_kmax values {captured_kmax} "
        f"instead of default 50 when no diagnostics block is present."
    )


# ---------------------------------------------------------------------------
# Preservation test 2 — Property 4
#
# Finite floats are not altered by JSON serialisation
#
# Validates: Requirements 3.3
#
# EXPECTED OUTCOME: PASSES on both unfixed and fixed code.
# Note: make_json_safe doesn't exist yet on unfixed code, so this test
# validates the baseline: json.dumps/json.loads round-trip for finite floats.
# ---------------------------------------------------------------------------

def test_make_json_safe_finite_values_unchanged(tmp_path):
    """
    **Preservation test 2** — Property 4: Preservation

    Validates: Requirements 3.3

    A dict of finite floats must round-trip through json.dumps / json.loads
    unchanged. This captures the baseline behaviour that must not regress
    after the Bug 2 fix (make_json_safe must not alter finite floats).

    On unfixed code: json.dumps writes finite floats as numbers → PASSES.
    After fix: make_json_safe must leave finite floats unchanged → PASSES.
    """
    d = {
        "spectral_radius": 0.9,
        "input_scaling": 0.1,
        "val_nmse": 0.15,
        "test_nmse": 0.16,
        "mc_total": 45.0,
        "zero": 0.0,
        "negative": -1.5,
        "small": 1e-10,
        "large": 1e10,
    }

    # Baseline: json round-trip must preserve finite floats
    json_text = json.dumps(d)
    recovered = json.loads(json_text)

    assert recovered == d, (
        f"REGRESSION: json round-trip altered finite float values. "
        f"Original: {d}, Recovered: {recovered}"
    )

    # After fix: make_json_safe (if available) must also leave finite floats unchanged
    try:
        from rc_lab.utils.io import make_json_safe
        safe = make_json_safe(d)
        assert safe == d, (
            f"REGRESSION: make_json_safe altered finite float values. "
            f"Original: {d}, After make_json_safe: {safe}"
        )
        # Also verify the round-trip still works after make_json_safe
        json_text_safe = json.dumps(safe)
        recovered_safe = json.loads(json_text_safe)
        assert recovered_safe == d, (
            f"REGRESSION: json round-trip after make_json_safe altered finite float values. "
            f"Original: {d}, Recovered: {recovered_safe}"
        )
    except ImportError:
        # make_json_safe not yet implemented — skip that part
        pass


# ---------------------------------------------------------------------------
# Preservation test 2 — Property 4 (property-based variant)
#
# For any dict of finite floats, json round-trip is identity.
# After fix: make_json_safe(d) == d for all-finite dicts.
#
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

_finite_float = st.floats(
    allow_nan=False,
    allow_infinity=False,
    min_value=-1e15,
    max_value=1e15,
)

_finite_float_dict = st.dictionaries(
    keys=st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=48), min_size=1, max_size=10),
    values=_finite_float,
    min_size=1,
    max_size=10,
)


@given(d=_finite_float_dict)
@settings(max_examples=50, deadline=10_000)
def test_make_json_safe_finite_values_unchanged_property(d):
    """
    **Property 4: Preservation** (property-based)

    Validates: Requirements 3.3

    For any dict of finite floats, json.dumps / json.loads round-trip must
    return an equal dict. This is the baseline behaviour.

    After fix: make_json_safe(d) must equal d for all-finite dicts.
    """
    # Baseline: json round-trip preserves finite floats
    json_text = json.dumps(d)
    recovered = json.loads(json_text)
    assert recovered == d, (
        f"json round-trip altered finite float values. "
        f"Original: {d}, Recovered: {recovered}"
    )

    # After fix: make_json_safe must also leave finite floats unchanged
    try:
        from rc_lab.utils.io import make_json_safe
        safe = make_json_safe(d)
        assert safe == d, (
            f"make_json_safe altered finite float values. "
            f"Original: {d}, After make_json_safe: {safe}"
        )
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Preservation test 3 — Property 6
#
# Full column set when all tasks enabled
#
# Validates: Requirements 3.2, 3.5, 3.6
#
# EXPECTED OUTCOME: PASSES on both unfixed and fixed code.
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
    **Preservation test 3** — Property 6: Preservation

    Validates: Requirements 3.2, 3.5, 3.6

    When all three tasks are enabled (enabled_tasks=["narma10", "mackey_glass",
    "memory_capacity"]) and mg_primary_metric="nmse", save_multitask_summary
    must produce a summary.csv with the full expected column set.

    This test MUST PASS on unfixed code (baseline behaviour) and MUST CONTINUE
    TO PASS after the Bug 3 fix is applied (no regression).
    """
    summary = _make_multitask_summary_all_tasks(tmp_path)
    save_multitask_summary(summary, tmp_path)

    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists(), f"summary.csv not found at {csv_path}"

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

    missing = _EXPECTED_FULL_COLUMNS - fieldnames
    assert not missing, (
        f"REGRESSION: summary.csv is missing expected columns when all tasks are enabled: "
        f"{sorted(missing)}. "
        f"Present columns: {sorted(fieldnames)}"
    )

    # Also verify no None-derived column names (sanity check)
    none_columns = [col for col in fieldnames if "None" in col]
    assert not none_columns, (
        f"REGRESSION: summary.csv contains columns with 'None' in their name "
        f"even when all tasks are enabled: {none_columns}"
    )


# ---------------------------------------------------------------------------
# Preservation test 4 — Backward compatibility
#
# Old MultiTaskSweepSummary objects without enabled_tasks field should produce
# the full column set (backward compat via getattr fallback).
#
# Validates: Requirements 2.9
#
# EXPECTED OUTCOME: PASSES on unfixed code (unfixed code doesn't check
# enabled_tasks at all, so it always produces the full column set).
# After fix: the getattr fallback must also produce the full column set.
# ---------------------------------------------------------------------------

def test_save_multitask_summary_backward_compat_no_enabled_tasks(tmp_path):
    """
    **Preservation test 4** — Backward compatibility

    Validates: Requirements 2.9

    Old MultiTaskSweepSummary objects without the enabled_tasks attribute
    should produce the full column set, preserving backward compatibility.

    On unfixed code: save_multitask_summary doesn't check enabled_tasks at all,
    so it always produces the full column set unconditionally → PASSES.

    After fix: the getattr fallback (default to all three tasks when
    enabled_tasks is absent) must also produce the full column set → PASSES.

    Implementation note: we cannot delete a dataclass field from an instance
    because dataclasses.asdict (called inside save_multitask_summary) requires
    the object to be a proper dataclass instance with all fields present.
    Instead, we verify the backward compat contract by:
      1. Confirming that getattr(summary, "enabled_tasks", default) returns
         the default when the attribute is absent — tested via a plain object.
      2. Confirming that a summary with enabled_tasks equal to the default
         (all three tasks) produces the full column set — this is what the
         getattr fallback will produce after the fix.
    """
    # Part 1: verify the getattr fallback contract works as expected.
    # This is the mechanism the fix will use for backward compat.
    import types
    old_obj = types.SimpleNamespace(
        sweep_name="old",
        n_configs=1,
        n_seeds=1,
        # No enabled_tasks attribute
    )
    fallback = getattr(old_obj, "enabled_tasks", ["narma10", "mackey_glass", "memory_capacity"])
    assert fallback == ["narma10", "mackey_glass", "memory_capacity"], (
        "getattr fallback for enabled_tasks does not return the expected default"
    )

    # Part 2: verify that a summary with enabled_tasks=default produces full columns.
    # This is what the fixed code will produce when enabled_tasks is absent
    # (via the getattr fallback). On unfixed code, the full column set is always
    # produced regardless of enabled_tasks.
    summary = _make_multitask_summary_all_tasks(tmp_path)
    # Explicitly set enabled_tasks to the default (all three tasks) to simulate
    # what the getattr fallback will return for old objects.
    summary.enabled_tasks = ["narma10", "mackey_glass", "memory_capacity"]

    save_multitask_summary(summary, tmp_path)

    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists(), f"summary.csv not found at {csv_path}"

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

    missing = _EXPECTED_FULL_COLUMNS - fieldnames
    assert not missing, (
        f"REGRESSION: summary.csv is missing expected columns in the backward "
        f"compat case (enabled_tasks defaults to all three tasks): {sorted(missing)}. "
        f"Present columns: {sorted(fieldnames)}"
    )

    none_columns = [col for col in fieldnames if "None" in col]
    assert not none_columns, (
        f"REGRESSION: summary.csv contains columns with 'None' in their name "
        f"in the backward compat case: {none_columns}"
    )
