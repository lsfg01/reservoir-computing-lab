import csv
import json

import numpy as np
import pytest

from rc_lab.runners.design_comparison_runner import DesignComparisonRunner
from rc_lab.runners.runner import resolve_task
from rc_lab.tasks.mackey_glass import (
    MackeyGlassTask,
    _build_prediction_pairs,
    _generate_subsampled_series,
    _integrate_mackey_glass,
    generate_mackey_glass,
)


def test_defaults_are_backward_compatible_shapes_and_params():
    task = MackeyGlassTask()
    data = task.generate(n_train=20, n_val=10, n_test=12, washout=5, seed=42)

    assert task.prediction_horizon == 1
    assert task.sample_stride == 1
    assert task.discard_transient == 0
    assert task.initial_history == "constant"
    assert data.u_train.shape == (25, 1)
    assert data.y_train.shape == (25, 1)
    assert data.u_val.shape == (10, 1)
    assert data.y_val.shape == (10, 1)
    assert data.u_test.shape == (12, 1)
    assert data.y_test.shape == (12, 1)


def test_prediction_horizon_shifts_pairs_on_subsampled_series():
    series = np.arange(12, dtype=float)
    u, y = _build_prediction_pairs(series, prediction_horizon=5)

    assert u.shape == (7, 1)
    assert y.shape == (7, 1)
    np.testing.assert_array_equal(y[:, 0], u[:, 0] + 5)


def test_sample_stride_generates_enough_points_and_expected_shapes():
    u, y = generate_mackey_glass(
        30,
        seed=42,
        sample_stride=10,
        prediction_horizon=4,
    )

    assert u.shape == (30, 1)
    assert y.shape == (30, 1)


def test_random_uniform_history_is_seed_reproducible():
    kwargs = {
        "T": 40,
        "initial_history": "random_uniform",
        "history_low": 1.1,
        "history_high": 1.3,
    }

    u1, y1 = generate_mackey_glass(seed=123, **kwargs)
    u2, y2 = generate_mackey_glass(seed=123, **kwargs)
    u3, y3 = generate_mackey_glass(seed=456, **kwargs)

    np.testing.assert_array_equal(u1, u2)
    np.testing.assert_array_equal(y1, y2)
    assert not np.array_equal(u1, u3)
    assert not np.array_equal(y1, y3)


def test_constant_history_ignores_seed_for_backward_compatibility():
    u1, y1 = generate_mackey_glass(40, seed=123)
    u2, y2 = generate_mackey_glass(40, seed=456)

    np.testing.assert_array_equal(u1, u2)
    np.testing.assert_array_equal(y1, y2)


def test_discard_transient_happens_before_subsampling():
    common = {
        "seed": 42,
        "tau": 17,
        "dt": 0.1,
        "beta": 0.2,
        "gamma": 0.1,
        "n": 10,
        "initial_history": "constant",
        "history_value": 0.9,
        "history_low": 1.1,
        "history_high": 1.3,
    }
    n_pairs = 10
    prediction_horizon = 3
    sample_stride = 4
    discard_transient = 7
    required = n_pairs + prediction_horizon
    total_steps = discard_transient + sample_stride * (required - 1) + 1

    raw = _integrate_mackey_glass(total_steps=total_steps, **common)
    expected = raw[discard_transient::sample_stride][:required]
    actual = _generate_subsampled_series(
        n_pairs=n_pairs,
        prediction_horizon=prediction_horizon,
        sample_stride=sample_stride,
        discard_transient=discard_transient,
        **common,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"prediction_horizon": 0}, "prediction_horizon"),
        ({"sample_stride": 0}, "sample_stride"),
        ({"discard_transient": -1}, "discard_transient"),
        ({"initial_history": "bad"}, "initial_history"),
        (
            {
                "initial_history": "random_uniform",
                "history_low": 1.3,
                "history_high": 1.3,
            },
            "history_low",
        ),
    ],
)
def test_validations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MackeyGlassTask(**kwargs)


def test_reset_protocol_shapes_include_warmup_full_blocks():
    task = MackeyGlassTask(state_policy="reset", prediction_horizon=5, sample_stride=2)
    data = task.generate(n_train=30, n_val=11, n_test=13, washout=7, seed=42)

    assert data.u_train.shape == (37, 1)
    assert data.y_train.shape == (37, 1)
    assert data.u_val_full.shape == (18, 1)
    assert data.u_test_full.shape == (20, 1)
    assert data.u_val.shape == (11, 1)
    assert data.y_val.shape == (11, 1)
    assert data.u_test.shape == (13, 1)
    assert data.y_test.shape == (13, 1)


def test_carryover_protocol_shapes():
    task = MackeyGlassTask(state_policy="carryover", prediction_horizon=5, sample_stride=2)
    data = task.generate(n_train=30, n_val=11, n_test=13, washout=7, seed=42)

    assert data.u_train.shape == (37, 1)
    assert data.u_val_full is None
    assert data.u_test_full is None
    assert data.u_val.shape == (11, 1)
    assert data.u_test.shape == (13, 1)


def test_resolve_task_passes_new_mackey_glass_params():
    task = resolve_task(
        "mackey_glass",
        state_policy="carryover",
        task_cfg={
            "tau": 18,
            "dt": 0.2,
            "beta": 0.3,
            "gamma": 0.05,
            "n": 8,
            "prediction_horizon": 12,
            "sample_stride": 3,
            "discard_transient": 20,
            "initial_history": "random_uniform",
            "history_value": 0.8,
            "history_low": 1.0,
            "history_high": 1.4,
        },
    )

    assert isinstance(task, MackeyGlassTask)
    assert task._tau == 18
    assert task._dt == 0.2
    assert task._beta == 0.3
    assert task._gamma == 0.05
    assert task._n == 8
    assert task.prediction_horizon == 12
    assert task.sample_stride == 3
    assert task.discard_transient == 20
    assert task.initial_history == "random_uniform"
    assert task._state_policy == "carryover"


def test_smoke_design_comparison_with_only_mackey_glass_enabled(tmp_path):
    cfg = {
        "sweep": {
            "name": "smoke_mg_only",
            "output_dir": str(tmp_path / "smoke_mg_only"),
            "seeds": [42],
        },
        "designs": [
            {
                "name": "random_sparse_baseline",
                "reservoir": {
                    "type": "random_sparse",
                    "N": 12,
                    "sparsity": 0.8,
                    "bias_scaling": 0.0,
                },
            },
        ],
        "grid": {
            "spectral_radius": [0.9],
            "input_scaling": [0.1],
            "leak_rate": [1.0],
        },
        "tasks": {
            "narma10": {
                "enabled": False,
                "n_train": 40,
                "n_val": 15,
                "n_test": 15,
                "washout": 5,
                "state_policy": "reset",
            },
            "mackey_glass": {
                "enabled": True,
                "n_train": 40,
                "n_val": 15,
                "n_test": 15,
                "washout": 5,
                "state_policy": "reset",
                "initial_history": "random_uniform",
                "discard_transient": 20,
                "sample_stride": 2,
                "prediction_horizon": 4,
            },
            "memory_capacity": {
                "enabled": False,
                "washout": 5,
                "input_length": 100,
                "fit_fraction": 0.5,
                "kmax": 5,
                "ridge_param": 1e-6,
            },
        },
        "readout": {
            "type": "ridge",
            "features": "states",
            "ridge_candidates": [1e-6],
        },
        "metrics": ["nmse", "rmse"],
        "ranking": {
            "shortlist_top_n": 1,
            "narma10": {"metric": "nmse", "direction": "min"},
            "mackey_glass": {"metric": "nmse", "direction": "min"},
            "memory_capacity": {"metric": "mc_total", "direction": "max"},
        },
        "diagnostics": {"transient_kmax": 3},
    }

    table = DesignComparisonRunner(cfg).run()
    output_dir = tmp_path / "smoke_mg_only"
    csv_path = output_dir / "comparison_summary.csv"
    json_path = output_dir / "comparison_summary.json"

    assert table
    assert csv_path.exists()
    assert json_path.exists()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    assert "mg_val_nmse_mean" in fieldnames
    assert not [col for col in fieldnames if col.startswith("narma10_")]
    assert "mc_total_mean" not in fieldnames

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["enabled_tasks"] == ["mackey_glass"]
