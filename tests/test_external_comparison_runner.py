import json
from pathlib import Path

import pytest

from rc_lab.runners.external_comparison_runner import ExternalComparisonRunner, expand_grid
from rc_lab.runners.sweep_runner import make_config_id


def _minimal_external_config(tmp_path: Path, *, enable_narma10: bool = False) -> dict:
    tasks = {
        "delay_recall": {
            "enabled": True,
            "name": "delay_recall",
            "n_train": 40,
            "n_val": 20,
            "n_test": 20,
            "washout": 3,
            "kmax": 3,
            "input_low": -1.0,
            "input_high": 1.0,
            "state_policy": "reset",
        },
        "narma10": {
            "enabled": enable_narma10,
            "name": "narma10",
            "n_train": 40,
            "n_val": 20,
            "n_test": 20,
            "washout": 10,
            "state_policy": "reset",
        },
        "mackey_glass": {
            "enabled": False,
            "name": "mackey_glass",
            "n_train": 40,
            "n_val": 20,
            "n_test": 20,
            "washout": 10,
            "state_policy": "reset",
        },
    }
    return {
        "sweep": {
            "name": "external_smoke",
            "output_dir": str(tmp_path / "external_smoke"),
            "seeds": [42],
        },
        "comparison": {
            "n_candidates_per_model": 2,
            "allow_unequal_candidate_counts": False,
            "use_test_for_selection": False,
            "device": "cpu",
        },
        "models": [
            {
                "name": "random_sparse",
                "kind": "esn",
                "reservoir": {
                    "type": "random_sparse",
                    "N": 6,
                    "sparsity": 0.7,
                    "bias_scaling": 0.0,
                },
                "grid": {
                    "spectral_radius": [0.5, 0.8],
                    "input_scaling": [0.2],
                    "leak_rate": [1.0],
                    "ridge_param": [1.0e-6],
                },
            },
            {
                "name": "tapped_delay_ridge",
                "kind": "tapped_delay_ridge",
                "grid": {
                    "n_lags": [1, 3],
                    "ridge_param": [1.0e-6],
                    "feature_mode": ["raw"],
                },
            },
        ],
        "tasks": tasks,
        "readout": {
            "type": "ridge",
            "features": "states",
            "ridge_candidates": [1.0e-6],
        },
        "metrics": {
            "common": ["nmse"],
            "memory": ["memory_corr_total", "memory_eff_total"],
        },
        "ranking": {
            "delay_recall": {"metric": "memory_corr_total", "direction": "max"},
            "narma10": {"metric": "nmse", "direction": "min"},
            "mackey_glass": {"metric": "nmse", "direction": "min"},
        },
    }


def test_expand_grid_cartesian_product():
    grid = {"a": [1, 2], "b": ["x", "y"]}
    points = expand_grid(grid)
    assert points == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_fairness_validation_rejects_wrong_candidate_count(tmp_path):
    cfg = _minimal_external_config(tmp_path)
    cfg["models"][1]["grid"]["n_lags"] = [1]
    with pytest.raises(ValueError, match="expected 2"):
        ExternalComparisonRunner(cfg)


def test_external_comparison_smoke_persists_expected_structure(tmp_path):
    cfg = _minimal_external_config(tmp_path)
    table = ExternalComparisonRunner(cfg).run()
    out = Path(cfg["sweep"]["output_dir"])

    assert table
    assert (out / "comparison_summary.csv").exists()
    assert (out / "comparison_summary.json").exists()
    for model in ("random_sparse", "tapped_delay_ridge"):
        assert (out / model / "summary.csv").exists()
        assert (out / model / "summary.json").exists()
        assert (out / model / "delay_recall" / "runs").is_dir()
        assert (out / model / "delay_recall" / "summary.csv").exists()
        assert (out / model / "delay_recall" / "summary.json").exists()

    data = json.loads((out / "comparison_summary.json").read_text(encoding="utf-8"))
    assert data["enabled_tasks"] == ["delay_recall"]
    assert "best_overall" in data


def test_disabled_tasks_do_not_create_global_columns(tmp_path):
    cfg = _minimal_external_config(tmp_path, enable_narma10=False)
    ExternalComparisonRunner(cfg).run()
    out = Path(cfg["sweep"]["output_dir"])
    data = json.loads((out / "comparison_summary.json").read_text(encoding="utf-8"))

    assert data["table"]
    first = data["table"][0]
    assert "delay_recall_memory_corr_total_mean" in first
    assert "narma10_val_nmse_mean" not in first
    assert "mg_val_nmse_mean" not in first


def test_external_best_aggregate_gets_complete_test_metrics(tmp_path, monkeypatch):
    cfg = _minimal_external_config(tmp_path, enable_narma10=True)
    cfg["comparison"]["n_candidates_per_model"] = 4
    cfg["models"] = [
        {
            "name": "tapped_delay_ridge",
            "kind": "tapped_delay_ridge",
            "grid": {
                "n_lags": [1, 2, 3, 4],
                "ridge_param": [1.0e-6],
                "feature_mode": ["raw"],
            },
        }
    ]

    test_calls: list[tuple[str, int, str | None]] = []
    delay_scores = {1: 4.0, 2: 1.0, 3: 3.0, 4: 2.0}
    narma_scores = {1: 4.0, 2: 1.0, 3: 2.0, 4: 3.0}

    def fake_run_single(
        self,
        model,
        task_name,
        config_point,
        config_id,
        seed,
        metric_names,
        evaluate_test,
        final_evaluation_reason=None,
    ):
        n_lags = config_point["n_lags"]
        if task_name == "delay_recall":
            values = {
                "nmse": 1.0 / delay_scores[n_lags],
                "memory_corr_total": delay_scores[n_lags],
                "memory_eff_total": delay_scores[n_lags] / 10.0,
            }
        else:
            values = {"nmse": narma_scores[n_lags]}
        val_metrics = {metric: values[metric] for metric in metric_names}
        test_metrics = {
            metric: values[metric] + 0.123
            for metric in metric_names
        } if evaluate_test else {}
        if evaluate_test:
            test_calls.append((task_name, n_lags, final_evaluation_reason))
        return {
            "sweep_name": self._sweep_name,
            "model_name": model["name"],
            "model_kind": model["kind"],
            "task_name": task_name,
            "config_id": config_id,
            "seed": seed,
            "config_point": config_point,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "timing": {"total_s": 0.0},
            "metadata": {"n_total_params": 1, "n_trainable_params": 1},
            "evaluation_phase": "final_test" if evaluate_test else "validation",
            "final_evaluation_reason": final_evaluation_reason if evaluate_test else None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(ExternalComparisonRunner, "_run_external_single", fake_run_single)

    table = ExternalComparisonRunner(cfg).run()
    out = Path(cfg["sweep"]["output_dir"])
    aggregate_config = {"n_lags": 3, "ridge_param": 1.0e-6, "feature_mode": "raw"}
    aggregate_id = make_config_id(aggregate_config)

    model_summary = json.loads((out / "tapped_delay_ridge" / "summary.json").read_text(encoding="utf-8"))
    assert model_summary["best_config_id"] == aggregate_id
    assert model_summary["task_summaries"]["delay_recall"]["best_config_id"] != aggregate_id
    assert model_summary["task_summaries"]["narma10"]["best_config_id"] != aggregate_id

    aggregate_row = next(row for row in model_summary["table"] if row["config_id"] == aggregate_id)
    assert "delay_recall_test_memory_corr_total_mean" in aggregate_row
    assert "narma10_test_nmse_mean" in aggregate_row

    comparison = json.loads((out / "comparison_summary.json").read_text(encoding="utf-8"))
    assert comparison["best_overall"]["config_id"] == aggregate_id
    assert "delay_recall_test_memory_corr_total_mean" in comparison["best_overall"]
    assert "narma10_test_nmse_mean" in comparison["best_overall"]

    assert test_calls.count(("delay_recall", 3, "best_aggregate_within_model")) == 1
    assert test_calls.count(("narma10", 3, "best_aggregate_within_model")) == 1
    assert not any(call[1] == 3 and call[2] == "best_aggregate_global" for call in test_calls)
    assert table[0]["config_id"] == aggregate_id
