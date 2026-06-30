from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rc_lab.analysis.task_rankings import (
    build_task_rankings,
    load_comparison_summary,
    save_task_rankings,
)


def _design_payload() -> dict:
    return {
        "source": "comparison_summary.json",
        "enabled_tasks": ["narma10", "mackey_glass", "memory_capacity"],
        "table": [
            {
                "design_name": "cycle",
                "reservoir_type": "cycle",
                "config_id": "cfg_b",
                "spectral_radius": 0.9,
                "input_scaling": 0.2,
                "leak_rate": 1.0,
                "narma10_val_nmse_mean": 0.08,
                "narma10_test_nmse_mean": 0.09,
                "mg_val_nmse_mean": 0.24,
                "mg_test_nmse_mean": 0.25,
                "mc_total_mean": 40.0,
                "global_rank_narma10": 1,
                "global_rank_mg": 3,
                "global_rank_mc": 4,
                "global_aggregate_rank": 2.66,
                "aggregate_rank_within_design": 1.0,
                "rank_narma10_within_design": 1,
                "rank_mg_within_design": 2,
                "rank_mc_within_design": 2,
            },
            {
                "design_name": "random_sparse_baseline",
                "reservoir_type": "random_sparse",
                "config_id": "cfg_a",
                "spectral_radius": 0.9,
                "input_scaling": 0.1,
                "leak_rate": 1.0,
                "narma10_val_nmse_mean": 0.10,
                "narma10_test_nmse_mean": 0.11,
                "mg_val_nmse_mean": 0.20,
                "mg_test_nmse_mean": 0.21,
                "mc_total_mean": 50.0,
                "global_rank_narma10": 2,
                "global_rank_mg": 1,
                "global_rank_mc": 3,
                "global_aggregate_rank": 2.0,
                "aggregate_rank_within_design": 1.0,
                "rank_narma10_within_design": 1,
                "rank_mg_within_design": 1,
                "rank_mc_within_design": 2,
            },
            {
                "design_name": "cycle",
                "reservoir_type": "cycle",
                "config_id": "cfg_a",
                "spectral_radius": 0.9,
                "input_scaling": 0.1,
                "leak_rate": 1.0,
                "narma10_val_nmse_mean": 0.12,
                "narma10_test_nmse_mean": 0.13,
                "mg_val_nmse_mean": 0.18,
                "mg_test_nmse_mean": 0.19,
                "mc_total_mean": 55.0,
                "global_rank_narma10": 3,
                "global_rank_mg": 2,
                "global_rank_mc": 2,
                "global_aggregate_rank": 2.33,
                "aggregate_rank_within_design": 2.0,
                "rank_narma10_within_design": 2,
                "rank_mg_within_design": 1,
                "rank_mc_within_design": 1,
            },
            {
                "design_name": "random_sparse_baseline",
                "reservoir_type": "random_sparse",
                "config_id": "cfg_b",
                "spectral_radius": 0.9,
                "input_scaling": 0.2,
                "leak_rate": 1.0,
                "narma10_val_nmse_mean": 0.15,
                "narma10_test_nmse_mean": 0.16,
                "mg_val_nmse_mean": 0.26,
                "mg_test_nmse_mean": 0.27,
                "mc_total_mean": 60.0,
                "global_rank_narma10": 4,
                "global_rank_mg": 4,
                "global_rank_mc": 1,
                "global_aggregate_rank": 3.0,
                "aggregate_rank_within_design": 2.0,
                "rank_narma10_within_design": 2,
                "rank_mg_within_design": 2,
                "rank_mc_within_design": 1,
            },
        ],
    }


def _external_payload() -> dict:
    return {
        "enabled_tasks": ["delay_recall", "narma10", "mackey_glass"],
        "table": [
            {
                "model_name": "random_sparse",
                "model_kind": "esn",
                "config_id": "esn_a",
                "delay_recall_memory_eff_total_mean": 0.50,
                "delay_recall_test_memory_eff_total_mean": 0.45,
                "narma10_val_nmse_mean": 0.20,
                "narma10_test_nmse_mean": 0.22,
                "mg_val_nmse_mean": 0.30,
                "mg_test_nmse_mean": 0.31,
                "global_rank_delay_recall": 2,
                "global_rank_narma10": 2,
                "global_rank_mg": 2,
                "global_aggregate_rank": 2.0,
                "aggregate_rank_within_model": 1.0,
                "n_total_params_mean": 100.0,
                "n_trainable_params_mean": 10.0,
            },
            {
                "model_name": "tapped_delay",
                "model_kind": "tapped_delay_ridge",
                "config_id": "td_a",
                "delay_recall_memory_eff_total_mean": 0.80,
                "delay_recall_test_memory_eff_total_mean": 0.75,
                "narma10_val_nmse_mean": 0.25,
                "narma10_test_nmse_mean": 0.27,
                "mg_val_nmse_mean": 0.40,
                "mg_test_nmse_mean": 0.41,
                "global_rank_delay_recall": 1,
                "global_rank_narma10": 3,
                "global_rank_mg": 3,
                "global_aggregate_rank": 2.33,
                "aggregate_rank_within_model": 1.0,
                "n_total_params_mean": 20.0,
                "n_trainable_params_mean": 20.0,
            },
            {
                "model_name": "lstm",
                "model_kind": "torch_lstm",
                "config_id": "lstm_a",
                "delay_recall_memory_eff_total_mean": 0.30,
                "delay_recall_test_memory_eff_total_mean": 0.28,
                "narma10_val_nmse_mean": 0.10,
                "narma10_test_nmse_mean": 0.12,
                "mg_val_nmse_mean": 0.10,
                "mg_test_nmse_mean": 0.11,
                "global_rank_delay_recall": 3,
                "global_rank_narma10": 1,
                "global_rank_mg": 1,
                "global_aggregate_rank": 1.66,
                "aggregate_rank_within_model": 1.0,
                "n_total_params_mean": 1000.0,
                "n_trainable_params_mean": 1000.0,
            },
        ],
    }


def test_design_task_rankings_include_all_design_tasks():
    rankings = build_task_rankings(_design_payload(), top_n=2)

    assert rankings["comparison_kind"] == "design"
    assert rankings["group_key"] == "design_name"
    assert rankings["enabled_tasks"] == ["narma10", "mackey_glass", "memory_capacity"]
    assert rankings["top_n"] == 2


def test_external_task_rankings_include_delay_recall_and_predictive_tasks():
    payload = _external_payload()
    expected = {
        "random_sparse": (100, 10, 10),
        "tapped_delay": (200, 20, 20),
        "lstm": (3000, 1000, 1000),
    }
    for row in payload["table"]:
        delay, narma, mg = expected[row["model_name"]]
        row["delay_recall_n_total_params"] = delay
        row["delay_recall_n_trainable_params"] = delay
        row["narma10_n_total_params"] = narma
        row["narma10_n_trainable_params"] = narma
        row["mg_n_total_params"] = mg
        row["mg_n_trainable_params"] = mg

    rankings = build_task_rankings(payload)

    assert rankings["comparison_kind"] == "external"
    assert rankings["group_key"] == "model_name"
    assert rankings["enabled_tasks"] == ["delay_recall", "narma10", "mackey_glass"]
    assert rankings["best_by_task"]["delay_recall"]["model_name"] == "tapped_delay"
    assert rankings["best_by_task"]["delay_recall"]["n_trainable_params"] == 200
    assert rankings["best_by_task"]["narma10"]["n_trainable_params"] == 1000
    assert "n_trainable_params_mean" not in rankings["best_by_task"]["narma10"]


def test_min_ranking_is_recomputed_when_rank_column_missing():
    payload = _design_payload()
    for row in payload["table"]:
        row.pop("global_rank_narma10")

    rankings = build_task_rankings(payload)

    top = rankings["top_by_task"]["narma10"]
    assert [row["config_id"] for row in top[:2]] == ["cfg_b", "cfg_a"]
    assert [row["task_global_rank"] for row in top[:2]] == [1, 2]


def test_max_ranking_is_recomputed_when_rank_column_missing():
    payload = _design_payload()
    for row in payload["table"]:
        row.pop("global_rank_mc")

    rankings = build_task_rankings(payload)

    top = rankings["top_by_task"]["memory_capacity"]
    assert top[0]["config_id"] == "cfg_b"
    assert top[0]["design_name"] == "random_sparse_baseline"
    assert top[0]["task_global_rank"] == 1


def test_recomputed_ranking_uses_min_method_for_ties():
    payload = _design_payload()
    for row in payload["table"]:
        row.pop("global_rank_narma10")
    payload["table"][0]["narma10_val_nmse_mean"] = 0.10
    payload["table"][1]["narma10_val_nmse_mean"] = 0.10

    rankings = build_task_rankings(payload)

    tied = [
        row for row in rankings["top_by_task"]["narma10"]
        if row["narma10_val_nmse_mean"] == pytest.approx(0.10)
    ]
    assert {row["task_global_rank"] for row in tied} == {1}


def test_best_by_task_uses_existing_global_rank_columns_when_available():
    rankings = build_task_rankings(_design_payload())

    assert rankings["best_by_task"]["narma10"]["config_id"] == "cfg_b"
    assert rankings["best_by_task"]["mackey_glass"]["config_id"] == "cfg_a"
    assert rankings["best_by_task"]["memory_capacity"]["config_id"] == "cfg_b"
    assert rankings["best_by_task"]["memory_capacity"]["design_name"] == "random_sparse_baseline"


def test_best_by_task_and_group_returns_one_best_row_per_group():
    rankings = build_task_rankings(_design_payload())

    by_group = rankings["best_by_task_and_group"]["mackey_glass"]
    assert by_group["cycle"]["config_id"] == "cfg_a"
    assert by_group["random_sparse_baseline"]["config_id"] == "cfg_a"


def test_save_task_rankings_exports_json_summary_and_task_csvs(tmp_path: Path):
    paths = save_task_rankings(_design_payload(), tmp_path, top_n=2)

    assert paths["json"].exists()
    assert paths["summary_csv"].exists()
    assert paths["narma10_csv"].exists()
    assert paths["mackey_glass_csv"].exists()
    assert paths["memory_capacity_csv"].exists()

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["top_n"] == 2
    with open(paths["summary_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["task"] for row in rows} == {"narma10", "mackey_glass", "memory_capacity"}


def test_disabled_task_is_skipped():
    payload = _design_payload()
    payload["enabled_tasks"] = ["narma10"]

    rankings = build_task_rankings(payload)

    assert rankings["enabled_tasks"] == ["narma10"]
    assert set(rankings["best_by_task"]) == {"narma10"}


def test_missing_optional_columns_are_omitted_from_csv(tmp_path: Path):
    paths = save_task_rankings(_external_payload(), tmp_path)

    with open(paths["delay_recall_csv"], newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        assert "diag_spectral_radius_mean" not in reader.fieldnames
        rows = list(reader)

    assert rows[0]["model_name"] == "tapped_delay"


def test_load_comparison_summary_records_source_filename(tmp_path: Path):
    path = tmp_path / "comparison_summary.json"
    path.write_text(json.dumps(_design_payload()), encoding="utf-8")

    payload = load_comparison_summary(path)
    rankings = build_task_rankings(payload)

    assert rankings["source"] == "comparison_summary.json"
