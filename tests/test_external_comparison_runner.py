import json
from pathlib import Path

import pytest

from rc_lab.runners.external_comparison_runner import ExternalComparisonRunner, expand_grid
from rc_lab.runners.sweep_runner import SweepRunResult, make_config_id
from rc_lab.utils.aggregation import aggregate_sweep_results


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


@pytest.mark.parametrize("candidate_style", ["grid", "candidates"])
def test_external_esn_candidate_ridge_param_rejected_with_readout_candidates(
    tmp_path,
    candidate_style,
):
    cfg = _minimal_external_config(tmp_path)
    esn_model = cfg["models"][0]
    if candidate_style == "grid":
        esn_model["grid"]["ridge_param"] = [1.0e-6]
    else:
        esn_model.pop("grid")
        esn_model["candidates"] = [
            {
                "spectral_radius": 0.5,
                "input_scaling": 0.2,
                "leak_rate": 1.0,
                "ridge_param": 1.0e-6,
            }
        ]

    with pytest.raises(ValueError, match="readout\\.ridge_candidates"):
        ExternalComparisonRunner(cfg)


def test_external_esn_candidates_direct_points_are_counted(tmp_path):
    cfg = _minimal_external_config(tmp_path)
    esn_model = cfg["models"][0]
    esn_model.pop("grid")
    esn_model["candidates"] = [
        {"spectral_radius": 0.5, "input_scaling": 0.2, "leak_rate": 1.0},
        {"spectral_radius": 0.8, "input_scaling": 0.2, "leak_rate": 1.0},
    ]

    payload = ExternalComparisonRunner(cfg).dry_run()

    row = next(model for model in payload["models"] if model["name"] == "random_sparse")
    assert row["n_candidates"] == 2
    assert row["candidate_spec"] == esn_model["candidates"]


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
    assert data["best_by_model"]["random_sparse"]["selected_total_s_mean"] is not None
    assert data["best_by_model"]["random_sparse"]["selected_total_s_std"] is not None
    assert data["best_by_model"]["random_sparse"]["tuning_total_s_sum"] is not None
    assert data["best_by_model"]["tapped_delay_ridge"]["selected_total_s_mean"] is not None
    assert data["best_by_model"]["tapped_delay_ridge"]["tuning_total_s_sum"] is not None

    esn_task_summary = json.loads((out / "random_sparse" / "delay_recall" / "summary.json").read_text(encoding="utf-8"))
    assert esn_task_summary["selected_total_s_mean"] is not None
    assert esn_task_summary["tuning_total_s_sum"] is not None


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
    assert comparison["best_overall"]["selected_total_s_mean"] is not None
    assert comparison["best_overall"]["tuning_total_s_sum"] is not None

    assert test_calls.count(("delay_recall", 3, "best_aggregate_within_model")) == 1
    assert test_calls.count(("narma10", 3, "best_aggregate_within_model")) == 1
    assert not any(call[1] == 3 and call[2] == "best_aggregate_global" for call in test_calls)
    assert table[0]["config_id"] == aggregate_id


def test_missing_run_timings_are_ignored_in_aggregation(tmp_path, monkeypatch):
    cfg = _minimal_external_config(tmp_path)
    cfg["sweep"]["seeds"] = [1, 2]
    cfg["models"] = [
        {
            "name": "tapped_delay_ridge",
            "kind": "tapped_delay_ridge",
            "grid": {
                "n_lags": [1, 2],
                "ridge_param": [1.0e-6],
                "feature_mode": ["raw"],
            },
        }
    ]

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
        score = 10.0 if config_point["n_lags"] == 1 else 5.0
        values = {
            "nmse": 1.0 / score,
            "memory_corr_total": score,
            "memory_eff_total": score,
        }
        return {
            "sweep_name": self._sweep_name,
            "model_name": model["name"],
            "model_kind": model["kind"],
            "task_name": task_name,
            "config_id": config_id,
            "seed": seed,
            "config_point": config_point,
            "val_metrics": {metric: values[metric] for metric in metric_names},
            "test_metrics": {metric: values[metric] for metric in metric_names} if evaluate_test else {},
            "timing": {"total_s": 2.0} if seed == 1 else {},
            "metadata": {"n_total_params": 1, "n_trainable_params": 1},
            "evaluation_phase": "final_test" if evaluate_test else "validation",
            "final_evaluation_reason": final_evaluation_reason if evaluate_test else None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(ExternalComparisonRunner, "_run_external_single", fake_run_single)
    ExternalComparisonRunner(cfg).run()

    out = Path(cfg["sweep"]["output_dir"])
    comparison = json.loads((out / "comparison_summary.json").read_text(encoding="utf-8"))
    best = comparison["best_by_model"]["tapped_delay_ridge"]
    assert best["selected_total_s_mean"] == 2.0
    assert best["selected_total_s_std"] == 0.0
    assert best["tuning_total_s_sum"] > 0.0


def test_external_tuning_total_is_preserved_when_final_run_overwrites_selected(tmp_path, monkeypatch):
    cfg = _minimal_external_config(tmp_path)
    cfg["sweep"]["seeds"] = [1]
    cfg["models"] = [
        {
            "name": "tapped_delay_ridge",
            "kind": "tapped_delay_ridge",
            "grid": {
                "n_lags": [1, 2],
                "ridge_param": [1.0e-6],
                "feature_mode": ["raw"],
            },
        }
    ]

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
        score = 10.0 if n_lags == 1 else 5.0
        values = {"nmse": 1.0 / score, "memory_corr_total": score, "memory_eff_total": score}
        if evaluate_test:
            timing = {
                "fit_s": 10.0,
                "final_test_s": 100.0,
                "test_s": 100.0,
                "selected_total_s": 110.0,
                "tuning_s": 999.0,
                "total_s": 999.0,
            }
        else:
            timing = {
                "fit_s": float(n_lags),
                "validation_s": 0.5,
                "tuning_s": float(n_lags) + 0.5,
                "total_s": float(n_lags) + 0.5,
            }
        return {
            "sweep_name": self._sweep_name,
            "model_name": model["name"],
            "model_kind": model["kind"],
            "task_name": task_name,
            "config_id": config_id,
            "seed": seed,
            "config_point": config_point,
            "val_metrics": {metric: values[metric] for metric in metric_names},
            "test_metrics": {metric: values[metric] for metric in metric_names} if evaluate_test else {},
            "timing": timing,
            "metadata": {"n_total_params": 1, "n_trainable_params": 1},
            "evaluation_phase": "final_test" if evaluate_test else "validation",
            "final_evaluation_reason": final_evaluation_reason if evaluate_test else None,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(ExternalComparisonRunner, "_run_external_single", fake_run_single)
    ExternalComparisonRunner(cfg).run()

    out = Path(cfg["sweep"]["output_dir"])
    comparison = json.loads((out / "comparison_summary.json").read_text(encoding="utf-8"))
    best = comparison["best_by_model"]["tapped_delay_ridge"]
    assert best["tuning_total_s_sum"] == pytest.approx(4.0)
    assert best["selected_fit_s_mean"] == pytest.approx(10.0)
    assert best["selected_test_s_mean"] == pytest.approx(100.0)
    assert best["selected_total_s_mean"] == pytest.approx(110.0)
    assert best["diagnostic_total_s_sum"] == pytest.approx(1003.0)


def test_esn_common_timings_use_validation_tuning_not_raw_total():
    runs = [
        SweepRunResult(
            sweep_name="esn_timing",
            config_id="best",
            seed=1,
            config_point={"spectral_radius": 0.9},
            best_ridge=1.0e-6,
            val_curve={},
            val_metrics={"memory_corr_total": 2.0},
            test_metrics={"memory_corr_total": 2.1},
            timing={
                "fit_s": 4.0,
                "test_s": 1.0,
                "final_test_s": 1.0,
                "selected_total_s": 5.0,
                "tuning_s": 4.0,
                "total_s": 100.0,
            },
            timestamp="2026-01-01T00:00:00+00:00",
        ),
        SweepRunResult(
            sweep_name="esn_timing",
            config_id="other",
            seed=1,
            config_point={"spectral_radius": 0.7},
            best_ridge=1.0e-6,
            val_curve={},
            val_metrics={"memory_corr_total": 1.0},
            test_metrics={"memory_corr_total": 1.1},
            timing={
                "fit_s": 6.0,
                "test_s": 2.0,
                "final_test_s": 2.0,
                "selected_total_s": 8.0,
                "tuning_s": 6.0,
                "total_s": 200.0,
            },
            timestamp="2026-01-01T00:00:00+00:00",
        ),
    ]

    summary = aggregate_sweep_results(
        runs,
        sweep_name="esn_timing",
        task_name="delay_recall",
        primary_metric="memory_corr_total",
        primary_direction="max",
    )

    assert summary.best_config_id == "best"
    assert summary.tuning_total_s_sum == pytest.approx(10.0)
    assert summary.selected_fit_s_mean == pytest.approx(4.0)
    assert summary.selected_test_s_mean == pytest.approx(1.0)
    assert summary.selected_total_s_mean == pytest.approx(5.0)
    assert summary.diagnostic_total_s_sum == pytest.approx(300.0)


def test_main_external_config_tapped_delay_is_limited_and_counts_match():
    import yaml

    cfg = yaml.safe_load(Path("configs/prelim_study/external/esn_vs_rnn_lstm.yaml").read_text(encoding="utf-8"))
    ExternalComparisonRunner(cfg)

    target = cfg["comparison"]["n_candidates_per_model"]
    by_name = {model["name"]: model for model in cfg["models"]}
    tapped = by_name["tapped_delay_ridge"]
    assert 100 not in tapped["grid"]["n_lags"]
    assert len(expand_grid(tapped["grid"])) == target == 16

    for name in ("simple_rnn", "lstm"):
        grid = by_name[name]["grid"]
        assert len(expand_grid(grid)) == target
        assert grid["num_layers"] == [1]
        assert grid["training_mode"] == ["windowed"]
        assert grid["normalize_inputs"] == [True]
        assert grid["normalize_targets"] == [True]
