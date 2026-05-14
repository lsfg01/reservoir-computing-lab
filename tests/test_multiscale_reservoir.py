import csv
import json

import numpy as np
import pytest

from rc_lab.reservoirs.multiscale import MultiScaleReservoir
from rc_lab.runners.design_comparison_runner import DesignComparisonRunner
from rc_lab.runners.runner import resolve_reservoir


SPECTRAL_RADIUS = 0.9
INPUT_SCALING = 0.1
SEED = 42


def test_multiscale_two_scale_shapes():
    res = MultiScaleReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        preset="two_scale_random",
    )
    mats = res.build(N=50, n_inputs=3, seed=SEED)

    assert mats.W.shape == (50, 50)
    assert mats.Win.shape == (50, 3)
    assert mats.bias.shape == (50,)


def test_multiscale_three_scale_shapes_and_block_sizes():
    res = MultiScaleReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        preset="three_scale_random",
    )
    mats = res.build(N=100, n_inputs=1, seed=SEED)

    assert mats.W.shape == (100, 100)
    assert MultiScaleReservoir._block_sizes(100, res.block_specs) == [30, 30, 40]


def test_multiscale_preset_and_block_specs_are_mutually_exclusive():
    with pytest.raises(ValueError):
        MultiScaleReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            preset="two_scale_random",
            block_specs=[
                {
                    "name": "a",
                    "fraction": 1.0,
                    "block_type": "random_sparse",
                    "rho_factor": 1.0,
                }
            ],
        )


def test_multiscale_requires_preset_or_block_specs():
    with pytest.raises(ValueError):
        MultiScaleReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
        )


def test_multiscale_unknown_preset_raises():
    with pytest.raises(ValueError):
        MultiScaleReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            preset="unknown",
        )


def test_multiscale_unsupported_block_type_raises():
    with pytest.raises(ValueError, match="solo soporta 'random_sparse'"):
        MultiScaleReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            block_specs=[
                {
                    "name": "cycle",
                    "fraction": 1.0,
                    "block_type": "cycle_jump",
                    "rho_factor": 1.0,
                }
            ],
        )


def test_multiscale_explicit_random_sparse_block_specs_builds():
    res = MultiScaleReservoir(
        spectral_radius=SPECTRAL_RADIUS,
        input_scaling=INPUT_SCALING,
        block_specs=[
            {
                "name": "fast",
                "fraction": 0.25,
                "block_type": "random_sparse",
                "rho_factor": 0.5,
                "sparsity": 0.8,
            },
            {
                "name": "medium",
                "fraction": 0.25,
                "block_type": "random_sparse",
                "rho_factor": 0.8,
            },
            {
                "name": "slow",
                "fraction": 0.5,
                "block_type": "random_sparse",
                "rho_factor": 1.0,
                "sparsity": 0.9,
            },
        ],
    )
    mats = res.build(N=40, n_inputs=2, seed=SEED)

    assert mats.W.shape == (40, 40)
    assert mats.Win.shape == (40, 2)
    assert mats.bias.shape == (40,)
    assert res.block_specs[1]["sparsity"] == pytest.approx(0.9)


def test_multiscale_invalid_fraction_sum_raises():
    with pytest.raises(ValueError, match="sumar 1.0"):
        MultiScaleReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            block_specs=[
                {
                    "name": "a",
                    "fraction": 0.4,
                    "block_type": "random_sparse",
                    "rho_factor": 1.0,
                },
                {
                    "name": "b",
                    "fraction": 0.4,
                    "block_type": "random_sparse",
                    "rho_factor": 1.0,
                },
            ],
        )


def test_multiscale_non_positive_fraction_raises():
    with pytest.raises(ValueError, match="fraction debe ser > 0"):
        MultiScaleReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            block_specs=[
                {
                    "name": "a",
                    "fraction": 0.0,
                    "block_type": "random_sparse",
                    "rho_factor": 1.0,
                },
                {
                    "name": "b",
                    "fraction": 1.0,
                    "block_type": "random_sparse",
                    "rho_factor": 1.0,
                },
            ],
        )


def test_multiscale_invalid_coupling_direction_raises():
    with pytest.raises(ValueError):
        MultiScaleReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            preset="two_scale_random",
            coupling_direction="sideways",
        )


def test_multiscale_invalid_coupling_mode_raises():
    with pytest.raises(ValueError):
        MultiScaleReservoir(
            spectral_radius=SPECTRAL_RADIUS,
            input_scaling=INPUT_SCALING,
            preset="two_scale_random",
            coupling_mode="all_to_all",
        )


def test_multiscale_reproducibility():
    cfg = {
        "spectral_radius": SPECTRAL_RADIUS,
        "input_scaling": INPUT_SCALING,
        "preset": "three_scale_random",
        "sparsity": 0.9,
        "coupling_strength": 0.02,
        "coupling_density": 0.1,
        "bias_scaling": 0.05,
    }
    mats1 = MultiScaleReservoir(**cfg).build(N=60, n_inputs=2, seed=SEED)
    mats2 = MultiScaleReservoir(**cfg).build(N=60, n_inputs=2, seed=SEED)

    np.testing.assert_array_equal(mats1.W, mats2.W)
    np.testing.assert_array_equal(mats1.Win, mats2.Win)
    np.testing.assert_array_equal(mats1.bias, mats2.bias)


def test_resolve_reservoir_multiscale():
    res = resolve_reservoir(
        {
            "type": "multiscale",
            "spectral_radius": SPECTRAL_RADIUS,
            "input_scaling": INPUT_SCALING,
            "preset": "two_scale_random",
        }
    )

    assert isinstance(res, MultiScaleReservoir)


def test_smoke_multiscale_design_comparison(tmp_path):
    cfg = {
        "sweep": {
            "name": "smoke_multiscale",
            "output_dir": str(tmp_path / "smoke_multiscale"),
            "seeds": [42],
        },
        "designs": [
            {
                "name": "random_sparse_baseline",
                "reservoir": {
                    "type": "random_sparse",
                    "N": 20,
                    "sparsity": 0.9,
                    "bias_scaling": 0.0,
                },
            },
            {
                "name": "multiscale_three_random",
                "reservoir": {
                    "type": "multiscale",
                    "N": 20,
                    "preset": "three_scale_random",
                    "sparsity": 0.9,
                    "coupling_strength": 0.02,
                    "coupling_density": 0.1,
                    "coupling_mode": "adjacent",
                    "coupling_direction": "bidirectional",
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
                "enabled": True,
                "n_train": 200,
                "n_val": 50,
                "n_test": 50,
                "washout": 20,
                "state_policy": "reset",
            },
            "mackey_glass": {
                "enabled": False,
                "n_train": 200,
                "n_val": 50,
                "n_test": 50,
                "washout": 20,
                "state_policy": "reset",
                "tau": 17,
                "dt": 0.1,
            },
            "memory_capacity": {
                "enabled": True,
                "washout": 20,
                "input_length": 200,
                "fit_fraction": 0.5,
                "kmax": 10,
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
            "shortlist_top_n": 2,
            "narma10": {"metric": "nmse", "direction": "min"},
            "mackey_glass": {"metric": "nmse", "direction": "min"},
            "memory_capacity": {"metric": "mc_total", "direction": "max"},
        },
        "diagnostics": {"transient_kmax": 5},
    }

    table = DesignComparisonRunner(cfg).run()
    output_dir = tmp_path / "smoke_multiscale"
    csv_path = output_dir / "comparison_summary.csv"
    json_path = output_dir / "comparison_summary.json"

    assert table
    assert csv_path.exists()
    assert json_path.exists()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    assert not [col for col in fieldnames if col.startswith("mg_")]

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["enabled_tasks"] == ["narma10", "memory_capacity"]
