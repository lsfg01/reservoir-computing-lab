from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from rc_lab.utils.io import make_json_safe


TASK_SPECS_DESIGN: dict[str, dict[str, Any]] = {
    "narma10": {
        "rank_col": "global_rank_narma10",
        "metric_col": "narma10_val_nmse_mean",
        "test_metric_col": "narma10_test_nmse_mean",
        "direction": "min",
    },
    "mackey_glass": {
        "rank_col": "global_rank_mg",
        "metric_col": "mg_val_nmse_mean",
        "test_metric_col": "mg_test_nmse_mean",
        "direction": "min",
    },
    "memory_capacity": {
        "rank_col": "global_rank_mc",
        "metric_col": "mc_total_mean",
        "test_metric_col": None,
        "direction": "max",
    },
}

TASK_SPECS_EXTERNAL: dict[str, dict[str, Any]] = {
    "delay_recall": {
        "rank_col": "global_rank_delay_recall",
        "metric_candidates": [
            "delay_recall_memory_eff_total_mean",
            "delay_recall_memory_corr_total_mean",
        ],
        "test_metric_candidates": [
            "delay_recall_test_memory_eff_total_mean",
            "delay_recall_test_memory_corr_total_mean",
        ],
        "direction": "max",
    },
    "narma10": {
        "rank_col": "global_rank_narma10",
        "metric_col": "narma10_val_nmse_mean",
        "test_metric_col": "narma10_test_nmse_mean",
        "direction": "min",
    },
    "mackey_glass": {
        "rank_col": "global_rank_mg",
        "metric_col": "mg_val_nmse_mean",
        "test_metric_col": "mg_test_nmse_mean",
        "direction": "min",
    },
}

DESIGN_TASK_COLUMNS = [
    "task",
    "task_global_rank",
    "design_name",
    "reservoir_type",
    "config_id",
    "spectral_radius",
    "input_scaling",
    "leak_rate",
    "task_val_metric",
    "task_test_metric",
    "global_aggregate_rank",
    "aggregate_rank_within_design",
    "rank_narma10_within_design",
    "rank_mg_within_design",
    "rank_mc_within_design",
    "global_rank_narma10",
    "global_rank_mg",
    "global_rank_mc",
    "diag_spectral_radius_mean",
    "diag_spectral_norm_mean",
    "diag_henrici_departure_mean",
    "diag_transient_growth_max_mean",
]

EXTERNAL_TASK_COLUMNS = [
    "task",
    "task_global_rank",
    "model_name",
    "model_kind",
    "config_id",
    "task_val_metric",
    "task_test_metric",
    "global_aggregate_rank",
    "aggregate_rank_within_model",
    "global_rank_delay_recall",
    "global_rank_narma10",
    "global_rank_mg",
    "n_total_params",
    "n_trainable_params",
    "selected_fit_s_mean",
    "selected_test_s_mean",
    "selected_total_s_mean",
    "tuning_total_s_sum",
]

SUMMARY_COLUMNS = [
    "task",
    "group_name",
    "group_kind",
    "task_global_rank",
    "config_id",
    "spectral_radius",
    "input_scaling",
    "leak_rate",
    "task_val_metric",
    "task_test_metric",
    "global_aggregate_rank",
]


def load_comparison_summary(path: str | Path) -> dict[str, Any]:
    """Load a persisted comparison_summary.json payload."""
    json_path = Path(path)
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload["_source"] = json_path.name
    return payload


def build_task_rankings(
    comparison_payload: dict[str, Any],
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """Build post-hoc global rankings per task from a comparison summary."""
    payload, _rows_by_task, _summary_rows = _build_outputs(comparison_payload, top_n=top_n)
    return payload


def save_task_rankings(
    comparison_payload: dict[str, Any],
    output_dir: str | Path,
    *,
    top_n: int = 20,
) -> dict[str, Path]:
    """Persist task ranking JSON and CSV artifacts under output_dir/task_rankings."""
    payload, rows_by_task, summary_rows = _build_outputs(comparison_payload, top_n=top_n)

    rankings_dir = Path(output_dir) / "task_rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    json_path = rankings_dir / "task_rankings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
    paths["json"] = json_path

    summary_path = rankings_dir / "task_rankings_summary.csv"
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    paths["summary_csv"] = summary_path

    task_columns = (
        DESIGN_TASK_COLUMNS
        if payload["comparison_kind"] == "design"
        else EXTERNAL_TASK_COLUMNS
    )
    for task_name, rows in rows_by_task.items():
        task_path = rankings_dir / f"{task_name}_global.csv"
        _write_csv(task_path, rows, task_columns)
        paths[f"{task_name}_csv"] = task_path

    return paths


def _build_outputs(
    comparison_payload: dict[str, Any],
    *,
    top_n: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = _comparison_table(comparison_payload)
    comparison_kind, group_key, group_label = _detect_comparison(rows)
    group_kind_key = "reservoir_type" if comparison_kind == "design" else "model_kind"
    task_specs = TASK_SPECS_DESIGN if comparison_kind == "design" else TASK_SPECS_EXTERNAL

    enabled_tasks = _enabled_tasks(comparison_payload, rows, task_specs)
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    best_by_task: dict[str, dict[str, Any]] = {}
    best_by_task_and_group: dict[str, dict[str, dict[str, Any]]] = {}
    top_by_task: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []

    for task_name in enabled_tasks:
        spec = task_specs[task_name]
        metric_col = _resolve_column(rows, spec, "metric")
        test_metric_col = _resolve_column(rows, spec, "test_metric")
        rank_col = spec["rank_col"]
        use_existing_rank = _has_column(rows, rank_col)

        if not use_existing_rank and metric_col is None:
            continue

        ranks = (
            [_rank_from_row(row, rank_col) for row in rows]
            if use_existing_rank
            else _compute_ranks(rows, metric_col, spec["direction"])
        )

        task_rows: list[dict[str, Any]] = []
        for row, rank in zip(rows, ranks, strict=True):
            task_row = dict(row)
            task_row["task"] = task_name
            task_row["task_global_rank"] = int(rank) if _is_finite(rank) else None
            task_row["task_val_metric"] = row.get(metric_col) if metric_col else None
            task_row["task_test_metric"] = row.get(test_metric_col) if test_metric_col else None
            if comparison_kind == "external":
                prefix = "mg" if task_name == "mackey_glass" else task_name
                task_row["n_total_params"] = row.get(
                    f"{prefix}_n_total_params",
                    row.get("n_total_params_mean"),
                )
                task_row["n_trainable_params"] = row.get(
                    f"{prefix}_n_trainable_params",
                    row.get("n_trainable_params_mean"),
                )
                task_row.pop("n_total_params_mean", None)
                task_row.pop("n_trainable_params_mean", None)
            task_rows.append(task_row)

        task_rows.sort(
            key=lambda row: (
                _rank_sort_value(row.get("task_global_rank")),
                _metric_sort_value(row.get("task_val_metric"), spec["direction"]),
                str(row.get(group_key, "")),
                str(row.get("config_id", "")),
            )
        )
        rows_by_task[task_name] = task_rows
        if task_rows:
            best_by_task[task_name] = task_rows[0]
            top_by_task[task_name] = task_rows[:top_n]

        by_group: dict[str, dict[str, Any]] = {}
        for task_row in task_rows:
            group_name = str(task_row.get(group_key, ""))
            if group_name and group_name not in by_group:
                by_group[group_name] = task_row
                summary_rows.append(_summary_row(task_row, group_name, group_kind_key))
        best_by_task_and_group[task_name] = by_group

    output_payload = {
        "source": str(
            comparison_payload.get("source")
            or comparison_payload.get("_source")
            or "comparison_summary.json"
        ),
        "comparison_kind": comparison_kind,
        "group_key": group_key,
        "group_label": group_label,
        "enabled_tasks": list(rows_by_task.keys()),
        "top_n": top_n,
        "best_by_task": best_by_task,
        "best_by_task_and_group": best_by_task_and_group,
        "top_by_task": top_by_task,
    }
    return output_payload, rows_by_task, summary_rows


def _comparison_table(comparison_payload: dict[str, Any]) -> list[dict[str, Any]]:
    table = comparison_payload.get("table")
    if not isinstance(table, list) or not table:
        raise ValueError("comparison_payload debe contener una tabla no vacia en 'table'.")
    if not all(isinstance(row, dict) for row in table):
        raise ValueError("comparison_payload['table'] debe ser una lista de dicts.")
    return table


def _detect_comparison(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    keys = set().union(*(row.keys() for row in rows))
    if "design_name" in keys:
        return "design", "design_name", "design"
    if "model_name" in keys:
        return "external", "model_name", "model"
    raise ValueError("No se pudo detectar comparison kind: falta design_name o model_name.")


def _enabled_tasks(
    comparison_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    task_specs: dict[str, dict[str, Any]],
) -> list[str]:
    requested = comparison_payload.get("enabled_tasks")
    candidates = [
        task_name for task_name in task_specs
        if requested is None or task_name in requested
    ]
    return [
        task_name for task_name in candidates
        if _task_has_data(rows, task_specs[task_name])
    ]


def _task_has_data(rows: list[dict[str, Any]], spec: dict[str, Any]) -> bool:
    return _has_column(rows, spec["rank_col"]) or _resolve_column(rows, spec, "metric") is not None


def _resolve_column(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    kind: str,
) -> str | None:
    direct_key = f"{kind}_col"
    candidates_key = f"{kind}_candidates"
    direct = spec.get(direct_key)
    if direct and _has_column(rows, direct):
        return direct
    for candidate in spec.get(candidates_key, []):
        if _has_column(rows, candidate):
            return candidate
    return None


def _has_column(rows: list[dict[str, Any]], column: str | None) -> bool:
    if column is None:
        return False
    return any(column in row for row in rows)


def _rank_from_row(row: dict[str, Any], rank_col: str) -> float:
    value = row.get(rank_col)
    return float(value) if _is_finite(value) else float("inf")


def _compute_ranks(
    rows: list[dict[str, Any]],
    metric_col: str,
    direction: str,
) -> list[int]:
    fill = float("inf") if direction == "min" else float("-inf")
    values = [
        float(row[metric_col])
        if _is_finite(row.get(metric_col))
        else fill
        for row in rows
    ]
    ordered = sorted(set(values), reverse=(direction == "max"))
    rank_by_value: dict[float, int] = {}
    position = 1
    for value in ordered:
        count = sum(1 for item in values if item == value)
        rank_by_value[value] = position
        position += count
    return [rank_by_value[value] for value in values]


def _summary_row(
    task_row: dict[str, Any],
    group_name: str,
    group_kind_key: str,
) -> dict[str, Any]:
    return {
        "task": task_row.get("task"),
        "group_name": group_name,
        "group_kind": task_row.get(group_kind_key),
        "task_global_rank": task_row.get("task_global_rank"),
        "config_id": task_row.get("config_id"),
        "spectral_radius": task_row.get("spectral_radius"),
        "input_scaling": task_row.get("input_scaling"),
        "leak_rate": task_row.get("leak_rate"),
        "task_val_metric": task_row.get("task_val_metric"),
        "task_test_metric": task_row.get("task_test_metric"),
        "global_aggregate_rank": task_row.get("global_aggregate_rank"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for key in preferred:
        if any(key in row and row.get(key) is not None for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    safe = make_json_safe(value)
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, ensure_ascii=False, allow_nan=False)
    return "" if safe is None else safe


def _rank_sort_value(value: Any) -> float:
    return float(value) if _is_finite(value) else float("inf")


def _metric_sort_value(value: Any, direction: str) -> float:
    if not _is_finite(value):
        return float("inf")
    metric = float(value)
    return metric if direction == "min" else -metric


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
