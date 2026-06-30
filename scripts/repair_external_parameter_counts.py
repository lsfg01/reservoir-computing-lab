"""Backfill task-specific parameter counts in a persisted external comparison.

This is a metadata-only repair: it reads the exact per-task counts already
stored in each model summary and propagates them to the model/comparison
tables. It does not train or evaluate any model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rc_lab.analysis.task_rankings import save_task_rankings
from rc_lab.runners.external_comparison_runner import _write_csv, _write_json


TASK_PREFIXES = {
    "delay_recall": "delay_recall",
    "narma10": "narma10",
    "mackey_glass": "mg",
}
PARAMETER_KEYS = [
    f"{prefix}_{suffix}"
    for prefix in TASK_PREFIXES.values()
    for suffix in ("n_total_params", "n_trainable_params")
]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _task_config(
    model_summary: dict[str, Any],
    task_name: str,
    config_id: str,
) -> dict[str, Any]:
    task_summary = model_summary.get("task_summaries", {}).get(task_name, {})
    for config in task_summary.get("configs", []):
        if config.get("config_id") == config_id:
            return config
    raise KeyError(
        f"Missing {task_name}/{config_id} in "
        f"{model_summary.get('model_name', '<unknown>')}"
    )


def _parameter_fields(
    model_summary: dict[str, Any],
    config_id: str,
) -> dict[str, float]:
    fields: dict[str, float] = {}
    for task_name, prefix in TASK_PREFIXES.items():
        config = _task_config(model_summary, task_name, config_id)
        metadata = config.get("metadata_mean", {})
        for suffix in ("n_total_params", "n_trainable_params"):
            value = metadata.get(suffix)
            if value is None:
                raise ValueError(
                    f"Missing metadata_mean.{suffix} for "
                    f"{model_summary.get('model_name')}/{task_name}/{config_id}"
                )
            fields[f"{prefix}_{suffix}"] = float(value)
    return fields


def _repair_model_summary(path: Path) -> dict[str, Any]:
    summary = _load_json(path)
    table = summary.get("table", [])
    if not isinstance(table, list):
        raise ValueError(f"Expected table list in {path}")

    by_config: dict[str, dict[str, Any]] = {}
    for row in table:
        config_id = str(row["config_id"])
        row.update(_parameter_fields(summary, config_id))
        by_config[config_id] = row

    best = summary.get("best_overall")
    if isinstance(best, dict) and best.get("config_id") in by_config:
        for key in PARAMETER_KEYS:
            best[key] = by_config[str(best["config_id"])][key]

    _write_json(path, summary)
    _write_csv(path.with_suffix(".csv"), table)
    return summary


def _copy_parameter_fields(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    for key in PARAMETER_KEYS:
        target[key] = source[key]


def repair_external_parameter_counts(results_dir: Path) -> dict[str, int]:
    comparison_path = results_dir / "comparison_summary.json"
    comparison = _load_json(comparison_path)

    repaired_models: dict[str, dict[str, Any]] = {}
    table_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for model_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        model_summary_path = model_dir / "summary.json"
        if not model_summary_path.exists():
            continue
        summary = _repair_model_summary(model_summary_path)
        model_name = str(summary["model_name"])
        repaired_models[model_name] = summary
        for row in summary["table"]:
            table_lookup[(model_name, str(row["config_id"]))] = row

    root_table = comparison.get("table", [])
    for row in root_table:
        key = (str(row["model_name"]), str(row["config_id"]))
        _copy_parameter_fields(row, table_lookup[key])

    best_overall = comparison.get("best_overall")
    if isinstance(best_overall, dict):
        key = (
            str(best_overall["model_name"]),
            str(best_overall["config_id"]),
        )
        _copy_parameter_fields(best_overall, table_lookup[key])

    best_by_model = comparison.get("best_by_model", {})
    if isinstance(best_by_model, dict):
        for row in best_by_model.values():
            key = (str(row["model_name"]), str(row["config_id"]))
            _copy_parameter_fields(row, table_lookup[key])

    _write_json(comparison_path, comparison)
    _write_csv(results_dir / "comparison_summary.csv", root_table)
    save_task_rankings(comparison, results_dir, top_n=20)

    return {
        "models": len(repaired_models),
        "candidate_rows": len(root_table),
        "task_specific_fields": len(root_table) * len(PARAMETER_KEYS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill exact per-task parameter counts without retraining"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing comparison_summary.json and model summaries",
    )
    args = parser.parse_args()

    stats = repair_external_parameter_counts(args.results_dir)
    print(
        "Parameter metadata repaired: "
        f"{stats['models']} models, "
        f"{stats['candidate_rows']} candidates, "
        f"{stats['task_specific_fields']} exact fields."
    )


if __name__ == "__main__":
    main()
