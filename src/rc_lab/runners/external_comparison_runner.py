from __future__ import annotations

import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from rc_lab.runners.runner import resolve_task
from rc_lab.runners.sweep_runner import (
    SweepSummary,
    make_config_id,
    validate_esn_ridge_location,
)
from rc_lab.sequence_models.tapped_delay import TappedDelayRidge
from rc_lab.sequence_models.training import compute_metrics
from rc_lab.tasks.base import TaskData
from rc_lab.utils.io import make_json_safe
from rc_lab.utils.seeding import set_seed
from rc_lab.utils.timing import timer


SUPPORTED_TASKS = {"delay_recall", "narma10", "mackey_glass"}
SUPPORTED_MODEL_KINDS = {"esn", "torch_simple_rnn", "torch_lstm", "tapped_delay_ridge", "narx_ridge", "ng_rc"}
PREDICTIVE_TASKS = {"narma10", "mackey_glass"}
MEMORY_METRICS = {
    "corr2_by_delay",
    "nmse_by_delay",
    "memory_corr_total",
    "memory_eff_total",
    "max_delay_corr_above_threshold",
}


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not isinstance(grid, dict) or not grid:
        raise ValueError("Each model grid must be a non-empty mapping")
    keys = list(grid.keys())
    values: list[list[Any]] = []
    for key in keys:
        raw = grid[key]
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"Grid entry {key!r} must be a non-empty list")
        values.append(raw)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def expand_model_candidates(model: dict[str, Any]) -> list[dict[str, Any]]:
    spec = _resolve_model_candidate_spec(model)
    if "candidates" in model:
        return _dedupe_candidate_points(spec, owner=f"Model {model['name']!r}")
    return expand_grid(spec)


def _resolve_model_candidate_spec(model: dict[str, Any]) -> Any:
    has_grid = "grid" in model
    has_candidates = "candidates" in model
    if has_grid == has_candidates:
        raise ValueError(
            f"Model {model.get('name', '<unnamed>')!r} must define exactly one "
            "of 'grid' or 'candidates'"
        )
    return model["candidates"] if has_candidates else model["grid"]


def _dedupe_candidate_points(candidates: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{owner} candidates must be a non-empty list")
    seen: set[str] = set()
    points: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"{owner} candidate at index {index} must be a non-empty mapping")
        point = dict(raw)
        config_id = make_config_id(point)
        if config_id in seen:
            continue
        seen.add(config_id)
        points.append(point)
    return points


def _candidate_grid_for_sweep(model: dict[str, Any]) -> dict[str, Any]:
    if "grid" in model:
        return {"grid": model["grid"]}
    points = expand_model_candidates(model)
    return {
        "grids": [
            {key: [value] for key, value in point.items()}
            for point in points
        ]
    }


class ExternalComparisonRunner:
    """
    Compare ESN/RC reservoirs with trainable sequence models.

    The on-disk layout intentionally mirrors DesignComparisonRunner:

        results/<comparison>/<model>/<task>/runs/
        results/<comparison>/<model>/<task>/summary.csv/json
        results/<comparison>/<model>/summary.csv/json
        results/<comparison>/comparison_summary.csv/json
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._validate_config(cfg)

        sweep = cfg["sweep"]
        self._sweep_name: str = sweep["name"]
        self._output_dir = Path(sweep["output_dir"])
        self._seeds: list[int] = list(sweep["seeds"])
        self._comparison: dict[str, Any] = cfg.get("comparison", {})
        self._models: list[dict[str, Any]] = [
            m for m in cfg["models"] if m.get("enabled", True)
        ]
        self._tasks: dict[str, Any] = cfg["tasks"]
        self._ranking: dict[str, Any] = cfg["ranking"]
        self._metrics_cfg: dict[str, Any] = cfg.get("metrics", {})
        self._device: str = self._comparison.get("device", "cpu")

    def dry_run(self) -> dict[str, Any]:
        active_tasks = self._enabled_tasks()
        model_rows = []
        total_validations = 0
        for model in self._models:
            n_candidates = len(expand_model_candidates(model))
            validations = n_candidates * len(self._seeds) * len(active_tasks)
            total_validations += validations
            row = {
                "name": model["name"],
                "kind": model["kind"],
                "n_candidates": n_candidates,
                "candidate_validations": validations,
                "candidate_spec": _resolve_model_candidate_spec(model),
            }
            if "grid" in model:
                row["grid"] = model["grid"]
            else:
                row["candidates"] = model["candidates"]
            model_rows.append(row)

        payload = {
            "sweep_name": self._sweep_name,
            "output_dir": str(self._output_dir),
            "enabled_tasks": active_tasks,
            "models": model_rows,
            "seeds": self._seeds,
            "total_candidate_validations": total_validations,
        }
        print(json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False))
        return payload

    def run(self) -> list[dict[str, Any]]:
        active_tasks = self._enabled_tasks()
        model_summaries: list[dict[str, Any]] = []

        for model in self._models:
            model_dir = self._output_dir / model["name"]
            task_summaries: dict[str, dict[str, Any]] = {}
            for task_name in active_tasks:
                if model["kind"] == "esn":
                    task_summary = self._run_esn_task(model, task_name, model_dir)
                else:
                    task_summary = self._run_external_task(model, task_name, model_dir)
                task_summaries[task_name] = task_summary

            model_summary = self._build_model_summary(model, task_summaries)
            if model["kind"] != "esn":
                model_summary = self._ensure_model_aggregate_tests(model, model_summary)
            self._save_model_summary(model_summary, model_dir)
            model_summaries.append(model_summary)

        comparison_table = self._build_comparison_summary(model_summaries)
        if self._ensure_global_aggregate_tests(comparison_table, model_summaries):
            comparison_table = self._build_comparison_summary(model_summaries)
        self._save_comparison_summary(comparison_table, model_summaries)
        return comparison_table

    def _validate_config(self, cfg: dict[str, Any]) -> None:
        for key in ("sweep", "comparison", "models", "tasks", "metrics", "ranking"):
            if key not in cfg:
                raise ValueError(f"External comparison config is missing required block {key!r}")

        sweep = cfg["sweep"]
        for key in ("name", "output_dir", "seeds"):
            if key not in sweep:
                raise ValueError(f"External comparison config is missing sweep.{key}")
        if not sweep["seeds"]:
            raise ValueError("sweep.seeds must be a non-empty list")

        comparison = cfg.get("comparison", {})
        if comparison.get("use_test_for_selection", False) is not False:
            raise ValueError("comparison.use_test_for_selection must be false")

        active_tasks = [
            name for name, task_cfg in cfg["tasks"].items()
            if task_cfg.get("enabled", True)
        ]
        if not active_tasks:
            raise ValueError("At least one task must be enabled")
        for task_name in active_tasks:
            if task_name not in SUPPORTED_TASKS:
                raise ValueError(f"Unsupported task {task_name!r}; supported: {sorted(SUPPORTED_TASKS)}")
            task_cfg = cfg["tasks"][task_name]
            if task_cfg.get("n_val", 0) <= 0:
                raise ValueError(f"Task {task_name!r} must define n_val > 0 for validation selection")
            if task_name == "delay_recall" and task_cfg.get("washout", 0) < task_cfg.get("kmax", 0):
                raise ValueError("delay_recall requires washout >= kmax")
            rank_cfg = cfg["ranking"].get(task_name)
            if not rank_cfg:
                raise ValueError(f"ranking.{task_name} must be defined for every active task")
            if rank_cfg.get("direction") not in {"min", "max"}:
                raise ValueError(f"ranking.{task_name}.direction must be 'min' or 'max'")
            metric = rank_cfg.get("metric")
            metric_name = str(metric).lower()
            if (
                not metric
                or metric_name.startswith("test")
                or metric_name.endswith("_test")
                or "_test_" in metric_name
            ):
                raise ValueError(f"ranking.{task_name}.metric must be a validation metric name, not test")

        models = cfg["models"]
        if not isinstance(models, list) or not models:
            raise ValueError("models must be a non-empty list")
        target_candidates = comparison.get("n_candidates_per_model")
        allow_unequal = comparison.get("allow_unequal_candidate_counts", False)
        readout_cfg = cfg.get("readout", {
            "type": "ridge",
            "features": "states",
            "ridge_candidates": [1e-6],
        })
        for idx, model in enumerate(models):
            if not model.get("enabled", True):
                continue
            if "name" not in model or "kind" not in model:
                raise ValueError(f"Model at index {idx} must have name and kind")
            if model["kind"] not in SUPPORTED_MODEL_KINDS:
                raise ValueError(f"Unsupported model kind {model['kind']!r}")
            if model["kind"] == "esn" and "reservoir" not in model:
                raise ValueError(f"ESN model {model['name']!r} must define reservoir")
            candidate_spec = _resolve_model_candidate_spec(model)
            if model["kind"] == "esn":
                validate_esn_ridge_location(
                    candidate_spec,
                    readout_cfg,
                    owner=f"ExternalComparisonRunner model {model['name']!r}",
                )
            n_candidates = len(expand_model_candidates(model))
            if target_candidates is not None and not allow_unequal and n_candidates != target_candidates:
                raise ValueError(
                    f"Model {model['name']!r} expands to {n_candidates} candidates, "
                    f"expected {target_candidates}"
                )

    def _enabled_tasks(self) -> list[str]:
        return [
            name for name in ("delay_recall", "narma10", "mackey_glass")
            if self._tasks.get(name, {}).get("enabled", False)
        ]

    def _metrics_for_task(self, task_name: str) -> list[str]:
        common = list(self._metrics_cfg.get("common", []))
        if task_name == "delay_recall":
            memory = list(self._metrics_cfg.get("memory", []))
            return _dedupe(common + memory)
        return common or ["nmse"]

    def _run_esn_task(
        self,
        model: dict[str, Any],
        task_name: str,
        model_dir: Path,
    ) -> dict[str, Any]:
        from rc_lab.runners.sweep_runner import SweepRunner

        ranking = self._ranking[task_name]
        sub_cfg: dict[str, Any] = {
            "sweep": {
                "name": f"{self._sweep_name}_{model['name']}_{task_name}",
                "output_dir": str(model_dir / task_name),
                "seeds": self._seeds,
            },
            "task": {
                "name": task_name,
                **self._tasks[task_name],
            },
            "reservoir": model["reservoir"],
            "readout": self._cfg.get("readout", {
                "type": "ridge",
                "features": "states",
                "ridge_candidates": [1e-6],
            }),
            "metrics": self._metrics_for_task(task_name),
            "primary_metric": ranking["metric"],
            "primary_direction": ranking["direction"],
        }
        sub_cfg.update(_candidate_grid_for_sweep(model))
        if "diagnostics" in self._cfg:
            sub_cfg["diagnostics"] = self._cfg["diagnostics"]

        summary = SweepRunner(sub_cfg).run()
        return self._task_summary_from_sweep(model, task_name, summary)

    def _run_external_task(
        self,
        model: dict[str, Any],
        task_name: str,
        model_dir: Path,
    ) -> dict[str, Any]:
        task_dir = model_dir / task_name
        runs_dir = task_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        config_points = expand_model_candidates(model)
        metric_names = self._metrics_for_task(task_name)
        runs_by_key: dict[tuple[str, int], dict[str, Any]] = {}

        for config_point in config_points:
            config_id = make_config_id(config_point)
            for seed in self._seeds:
                run = self._run_external_single(
                    model=model,
                    task_name=task_name,
                    config_point=config_point,
                    config_id=config_id,
                    seed=seed,
                    metric_names=metric_names,
                    evaluate_test=False,
                )
                runs_by_key[(config_id, seed)] = run
                self._save_run(run, runs_dir)

        validation_runs = list(runs_by_key.values())
        tuning_total_s_sum = _sum_run_timing(validation_runs, "tuning_s")
        diagnostic_total_s_sum = _sum_run_timing(validation_runs, "total_s")
        best_config_id = self._select_best_config(list(runs_by_key.values()), task_name)
        selected_point = next(cp for cp in config_points if make_config_id(cp) == best_config_id)

        for seed in self._seeds:
            run = self._run_external_single(
                model=model,
                task_name=task_name,
                config_point=selected_point,
                config_id=best_config_id,
                seed=seed,
                metric_names=metric_names,
                evaluate_test=True,
                final_evaluation_reason="best_by_task",
            )
            runs_by_key[(best_config_id, seed)] = run
            diagnostic_total_s_sum = _add_optional(
                diagnostic_total_s_sum,
                _timing_from_run(run, "total_s"),
            )
            self._save_run(run, runs_dir)

        summary = self._build_task_summary(
            model=model,
            task_name=task_name,
            runs=list(runs_by_key.values()),
            best_config_id=best_config_id,
            metric_names=metric_names,
            tuning_total_s_sum=tuning_total_s_sum,
            diagnostic_total_s_sum=diagnostic_total_s_sum,
        )
        self._save_task_summary(summary, task_dir)
        return summary

    def _ensure_model_aggregate_tests(
        self,
        model: dict[str, Any],
        model_summary: dict[str, Any],
    ) -> dict[str, Any]:
        best_config_id = model_summary.get("best_config_id")
        if not best_config_id:
            return model_summary
        changed = self._ensure_external_tests_for_config(
            model=model,
            task_summaries=model_summary["task_summaries"],
            config_id=best_config_id,
            final_evaluation_reason="best_aggregate_within_model",
        )
        if not changed:
            return model_summary
        return self._build_model_summary(model, model_summary["task_summaries"])

    def _ensure_global_aggregate_tests(
        self,
        comparison_table: list[dict[str, Any]],
        model_summaries: list[dict[str, Any]],
    ) -> bool:
        if not comparison_table:
            return False
        best = min(
            comparison_table,
            key=lambda row: row.get("aggregate_rank_global", row.get("global_aggregate_rank", float("inf"))),
        )
        model = self._model_by_name(best["model_name"])
        if model["kind"] == "esn":
            return False

        for i, model_summary in enumerate(model_summaries):
            if model_summary["model_name"] != best["model_name"]:
                continue
            changed = self._ensure_external_tests_for_config(
                model=model,
                task_summaries=model_summary["task_summaries"],
                config_id=best["config_id"],
                final_evaluation_reason="best_aggregate_global",
            )
            if not changed:
                return False
            new_summary = self._build_model_summary(model, model_summary["task_summaries"])
            model_summaries[i] = new_summary
            self._save_model_summary(new_summary, self._output_dir / model["name"])
            return True
        return False

    def _ensure_external_tests_for_config(
        self,
        model: dict[str, Any],
        task_summaries: dict[str, dict[str, Any]],
        config_id: str,
        final_evaluation_reason: str,
    ) -> bool:
        changed = False
        model_dir = self._output_dir / model["name"]

        for task_name, summary in list(task_summaries.items()):
            task_changed = False
            diagnostic_total_s_sum = summary.get("diagnostic_total_s_sum")
            cfg = _summary_config_by_id(summary, config_id)
            metric_names = self._metrics_for_task(task_name)
            runs_dir = model_dir / task_name / "runs"
            runs = self._load_task_runs(runs_dir)
            by_key = {
                (run["config_id"], run["seed"]): run
                for run in runs
            }

            for seed in self._seeds:
                key = (config_id, seed)
                existing = by_key.get(key)
                if existing is not None and _has_complete_test_metrics(existing, metric_names):
                    continue
                run = self._run_external_single(
                    model=model,
                    task_name=task_name,
                    config_point=cfg["config_point"],
                    config_id=config_id,
                    seed=seed,
                    metric_names=metric_names,
                    evaluate_test=True,
                    final_evaluation_reason=final_evaluation_reason,
                )
                by_key[key] = run
                self._save_run(run, runs_dir)
                diagnostic_total_s_sum = _add_optional(
                    diagnostic_total_s_sum,
                    _timing_from_run(run, "total_s"),
                )
                changed = True
                task_changed = True

            if task_changed:
                updated_runs = sorted(
                    by_key.values(),
                    key=lambda run: (run["config_id"], run["seed"]),
                )
                updated_summary = self._build_task_summary(
                    model=model,
                    task_name=task_name,
                    runs=updated_runs,
                    best_config_id=summary["best_config_id"],
                    metric_names=metric_names,
                    tuning_total_s_sum=summary.get("tuning_total_s_sum"),
                    diagnostic_total_s_sum=diagnostic_total_s_sum,
                )
                task_summaries[task_name] = updated_summary
                self._save_task_summary(updated_summary, model_dir / task_name)

        return changed

    def _load_task_runs(self, runs_dir: Path) -> list[dict[str, Any]]:
        if not runs_dir.exists():
            return []
        runs: list[dict[str, Any]] = []
        for path in sorted(runs_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        return runs

    def _model_by_name(self, model_name: str) -> dict[str, Any]:
        for model in self._models:
            if model["name"] == model_name:
                return model
        raise KeyError(model_name)

    def _run_external_single(
        self,
        model: dict[str, Any],
        task_name: str,
        config_point: dict[str, Any],
        config_id: str,
        seed: int,
        metric_names: list[str],
        evaluate_test: bool,
        final_evaluation_reason: str | None = None,
    ) -> dict[str, Any]:
        set_seed(seed)
        task_cfg = self._tasks[task_name]
        task = resolve_task(
            task_cfg.get("name", task_name),
            state_policy=task_cfg.get("state_policy", "reset"),
            task_cfg=task_cfg,
        )
        task_data = task.generate(
            n_train=task_cfg["n_train"],
            n_val=task_cfg["n_val"],
            n_test=task_cfg["n_test"],
            washout=task_cfg["washout"],
            seed=seed,
        )

        if model["kind"] in {"tapped_delay_ridge", "narx_ridge", "ng_rc"}:
            result = self._fit_tapped_delay(task_data, config_point, metric_names, evaluate_test)
        elif model["kind"] in {"torch_simple_rnn", "torch_lstm"}:
            from rc_lab.sequence_models.torch_models import fit_torch_sequence_model
            result = fit_torch_sequence_model(
                kind=model["kind"],
                task_data=task_data,
                config_point=config_point,
                metrics=metric_names,
                seed=seed,
                device=self._device,
                evaluate_test=evaluate_test,
                task_name=task_name,
                task_cfg=task_cfg,
            )
        else:
            raise ValueError(f"Unsupported external model kind: {model['kind']!r}")

        return {
            "sweep_name": self._sweep_name,
            "model_name": model["name"],
            "model_kind": model["kind"],
            "task_name": task_name,
            "config_id": config_id,
            "seed": seed,
            "config_point": config_point,
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "timing": result["timing"],
            "metadata": result["metadata"],
            "evaluation_phase": "final_test" if evaluate_test else "validation",
            "final_evaluation_reason": final_evaluation_reason if evaluate_test else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fit_tapped_delay(
        self,
        task_data: TaskData,
        config_point: dict[str, Any],
        metric_names: list[str],
        evaluate_test: bool,
    ) -> dict[str, Any]:
        model = TappedDelayRidge(
            n_lags=int(config_point.get("n_lags", 1)),
            ridge_param=float(config_point.get("ridge_param", 1e-6)),
            feature_mode=config_point.get("feature_mode", "raw"),
        )
        timing: dict[str, float] = {}
        with timer() as t_fit:
            model.fit(task_data.u_train, task_data.y_train, washout=task_data.washout)
        timing["train_s"] = t_fit["elapsed"]
        timing["fit_s"] = timing["train_s"]

        with timer() as t_val:
            val_split = self._prepare_tapped_val(model, task_data)
            y_pred_val = model.predict_prepared(val_split.features)
            val_metrics = compute_metrics(val_split.targets, y_pred_val, metric_names)
        timing["validation_s"] = t_val["elapsed"]

        if evaluate_test:
            with timer() as t_test:
                test_split = self._prepare_tapped_test(model, task_data)
                y_pred_test = model.predict_prepared(test_split.features)
                test_metrics = compute_metrics(test_split.targets, y_pred_test, metric_names)
            timing["final_test_s"] = t_test["elapsed"]
            timing["test_s"] = timing["final_test_s"]
        else:
            test_metrics = {}
        timing["eval_s"] = timing["validation_s"] + timing.get("final_test_s", 0.0)
        timing["tuning_s"] = timing["fit_s"] + timing["validation_s"]
        if evaluate_test:
            timing["selected_total_s"] = timing["fit_s"] + timing["final_test_s"]
        timing["total_s"] = timing["train_s"] + timing["eval_s"]

        return {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "timing": timing,
            "metadata": {
                "effective_washout": model.effective_washout,
                "n_total_params": model.n_total_params,
                "n_trainable_params": model.n_trainable_params,
            },
        }

    def _prepare_tapped_val(self, model: TappedDelayRidge, task_data: TaskData) -> Any:
        if task_data.y_val is None or task_data.u_val is None:
            raise ValueError("Validation split is required")
        if task_data.u_val_full is not None:
            return model.prepare_reset_scored_split(task_data.u_val_full, task_data.y_val, task_data.washout)
        return model.prepare_scored_with_history(task_data.u_train, task_data.u_val, task_data.y_val)

    def _prepare_tapped_test(self, model: TappedDelayRidge, task_data: TaskData) -> Any:
        if task_data.u_test_full is not None:
            return model.prepare_reset_scored_split(task_data.u_test_full, task_data.y_test, task_data.washout)
        if task_data.u_val is not None:
            history = np.vstack([task_data.u_train, task_data.u_val])
        else:
            history = task_data.u_train
        return model.prepare_scored_with_history(history, task_data.u_test, task_data.y_test)

    def _task_summary_from_sweep(
        self,
        model: dict[str, Any],
        task_name: str,
        summary: SweepSummary,
    ) -> dict[str, Any]:
        configs: list[dict[str, Any]] = []
        for cfg in summary.configs:
            configs.append({
                "config_id": cfg.config_id,
                "config_point": cfg.config_point,
                "n_seeds": cfg.n_seeds,
                "val_mean": cfg.val_mean,
                "val_std": cfg.val_std,
                "test_mean": cfg.test_mean,
                "test_std": cfg.test_std,
                "best_ridge_mode": cfg.best_ridge_mode,
                "timing_mean": dict(getattr(cfg, "timing_mean", {})),
                "timing_std": dict(getattr(cfg, "timing_std", {})),
                "timing_sum": dict(getattr(cfg, "timing_sum", {})),
                "metadata_mean": self._estimate_esn_metadata(model, task_name, cfg.config_point),
                "metadata_std": {},
            })

        return {
            "sweep_name": summary.sweep_name,
            "model_name": model["name"],
            "model_kind": model["kind"],
            "task_name": task_name,
            "n_configs": summary.n_configs,
            "n_seeds": summary.n_seeds,
            "selection_metric": self._ranking[task_name]["metric"],
            "selection_direction": self._ranking[task_name]["direction"],
            "configs": configs,
            "best_config_id": summary.best_config_id,
            "selected_fit_s_mean": summary.selected_fit_s_mean,
            "selected_fit_s_std": summary.selected_fit_s_std,
            "selected_test_s_mean": summary.selected_test_s_mean,
            "selected_test_s_std": summary.selected_test_s_std,
            "selected_total_s_mean": summary.selected_total_s_mean,
            "selected_total_s_std": summary.selected_total_s_std,
            "tuning_total_s_sum": summary.tuning_total_s_sum,
            "diagnostic_total_s_sum": summary.diagnostic_total_s_sum,
            "timestamp": summary.timestamp,
        }

    def _estimate_esn_metadata(
        self,
        model: dict[str, Any],
        task_name: str,
        config_point: dict[str, Any],
    ) -> dict[str, Any]:
        task_cfg = self._tasks[task_name]
        n_inputs = 1
        n_outputs = int(task_cfg.get("kmax", 1)) if task_name == "delay_recall" else 1
        n = int(model.get("reservoir", {}).get("N", 0))
        features = self._cfg.get("readout", {}).get("features", "states")
        readout_features = n if features == "states" else 1 + n_inputs + n
        reservoir_params = n * n + n * n_inputs + n
        readout_params = readout_features * n_outputs
        return {
            "n_total_params": int(reservoir_params + readout_params),
            "n_trainable_params": int(readout_params),
        }

    def _build_task_summary(
        self,
        model: dict[str, Any],
        task_name: str,
        runs: list[dict[str, Any]],
        best_config_id: str,
        metric_names: list[str],
        tuning_total_s_sum: float | None = None,
        diagnostic_total_s_sum: float | None = None,
    ) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            groups.setdefault(run["config_id"], []).append(run)

        configs = []
        for config_id, group in groups.items():
            val_keys = _metric_keys(group, "val_metrics", metric_names)
            test_keys = _metric_keys(group, "test_metrics", metric_names)
            configs.append({
                "config_id": config_id,
                "config_point": group[0]["config_point"],
                "n_seeds": len(group),
                "val_mean": {k: _mean([r["val_metrics"][k] for r in group if k in r["val_metrics"]]) for k in val_keys},
                "val_std": {k: _std([r["val_metrics"][k] for r in group if k in r["val_metrics"]]) for k in val_keys},
                "test_mean": {k: _mean([r["test_metrics"][k] for r in group if k in r["test_metrics"]]) for k in test_keys},
                "test_std": {k: _std([r["test_metrics"][k] for r in group if k in r["test_metrics"]]) for k in test_keys},
                "timing_mean": _aggregate_numeric_dict(group, "timing", np.mean),
                "timing_std": _aggregate_numeric_dict(group, "timing", np.std),
                "timing_sum": _aggregate_numeric_dict(group, "timing", np.sum),
                "metadata_mean": _aggregate_numeric_dict(group, "metadata", np.mean),
                "metadata_std": _aggregate_numeric_dict(group, "metadata", np.std),
            })

        best_cfg = next((cfg for cfg in configs if cfg["config_id"] == best_config_id), None)
        if tuning_total_s_sum is None:
            tuning_total_s_sum = _sum_run_timing(
                [run for run in runs if run.get("evaluation_phase") == "validation"],
                "tuning_s",
            )
        if diagnostic_total_s_sum is None:
            diagnostic_total_s_sum = _sum_run_timing(runs, "total_s")
        selected_times = _selected_time_fields(best_cfg)

        return {
            "sweep_name": f"{self._sweep_name}_{model['name']}_{task_name}",
            "model_name": model["name"],
            "model_kind": model["kind"],
            "task_name": task_name,
            "n_configs": len(configs),
            "n_seeds": len(self._seeds),
            "selection_metric": self._ranking[task_name]["metric"],
            "selection_direction": self._ranking[task_name]["direction"],
            "configs": configs,
            "best_config_id": best_config_id,
            **selected_times,
            "tuning_total_s_sum": tuning_total_s_sum,
            "diagnostic_total_s_sum": diagnostic_total_s_sum,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _select_best_config(self, runs: list[dict[str, Any]], task_name: str) -> str:
        metric = self._ranking[task_name]["metric"]
        direction = self._ranking[task_name]["direction"]
        grouped: dict[str, list[float]] = {}
        for run in runs:
            if metric not in run["val_metrics"]:
                raise ValueError(f"Metric {metric!r} not found in validation metrics for task {task_name!r}")
            value = _scalar(run["val_metrics"][metric])
            if not np.isfinite(value):
                value = float("inf") if direction == "min" else float("-inf")
            grouped.setdefault(run["config_id"], []).append(value)
        scores = {cid: float(np.mean(values)) for cid, values in grouped.items()}
        if direction == "min":
            return min(scores, key=scores.__getitem__)
        return max(scores, key=scores.__getitem__)

    def _build_model_summary(
        self,
        model: dict[str, Any],
        task_summaries: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        config_ids = sorted({
            cfg["config_id"]
            for summary in task_summaries.values()
            for cfg in summary["configs"]
        })
        rows = []
        for config_id in config_ids:
            row: dict[str, Any] = {
                "model_name": model["name"],
                "model_kind": model["kind"],
                "config_id": config_id,
                "n_seeds": len(self._seeds),
            }
            config_point = self._config_point_for(config_id, task_summaries)
            row.update(config_point)
            row["config_point"] = config_point

            total_times: list[float] = []
            selected_fit_means: list[float] = []
            selected_fit_vars: list[float] = []
            selected_test_means: list[float] = []
            selected_test_vars: list[float] = []
            selected_total_means: list[float] = []
            selected_total_vars: list[float] = []
            tuning_total_sums: list[float] = []
            diagnostic_total_sums: list[float] = []
            total_params: list[float] = []
            trainable_params: list[float] = []
            for task_name, summary in task_summaries.items():
                cfg = _summary_config_by_id(summary, config_id)
                self._add_task_columns(row, task_name, cfg)
                total_s = cfg.get("timing_mean", {}).get("total_s")
                if _is_finite_number(total_s):
                    total_times.append(float(total_s))
                selected_times = _selected_time_fields(cfg)
                fit_mean = selected_times["selected_fit_s_mean"]
                fit_std = selected_times["selected_fit_s_std"]
                test_mean = selected_times["selected_test_s_mean"]
                test_std = selected_times["selected_test_s_std"]
                selected_total = selected_times["selected_total_s_mean"]
                selected_total_std = selected_times["selected_total_s_std"]
                if _is_finite_number(fit_mean):
                    selected_fit_means.append(float(fit_mean))
                if _is_finite_number(fit_std):
                    selected_fit_vars.append(float(fit_std) ** 2)
                if _is_finite_number(test_mean):
                    selected_test_means.append(float(test_mean))
                if _is_finite_number(test_std):
                    selected_test_vars.append(float(test_std) ** 2)
                if _is_finite_number(selected_total):
                    selected_total_means.append(float(selected_total))
                if _is_finite_number(selected_total_std):
                    selected_total_vars.append(float(selected_total_std) ** 2)
                tuning_total_s = summary.get("tuning_total_s_sum")
                if _is_finite_number(tuning_total_s):
                    tuning_total_sums.append(float(tuning_total_s))
                diagnostic_total_s = summary.get("diagnostic_total_s_sum")
                if _is_finite_number(diagnostic_total_s):
                    diagnostic_total_sums.append(float(diagnostic_total_s))
                n_total = cfg.get("metadata_mean", {}).get("n_total_params")
                n_trainable = cfg.get("metadata_mean", {}).get("n_trainable_params")
                if n_total is not None:
                    total_params.append(float(n_total))
                if n_trainable is not None:
                    trainable_params.append(float(n_trainable))
            row["diagnostic_total_s_mean"] = float(np.mean(total_times)) if total_times else None
            row["selected_fit_s_mean"] = float(np.sum(selected_fit_means)) if selected_fit_means else None
            row["selected_fit_s_std"] = float(np.sqrt(np.sum(selected_fit_vars))) if selected_fit_vars else None
            row["selected_test_s_mean"] = float(np.sum(selected_test_means)) if selected_test_means else None
            row["selected_test_s_std"] = float(np.sqrt(np.sum(selected_test_vars))) if selected_test_vars else None
            row["selected_total_s_mean"] = float(np.sum(selected_total_means)) if selected_total_means else None
            row["selected_total_s_std"] = float(np.sqrt(np.sum(selected_total_vars))) if selected_total_vars else None
            row["tuning_total_s_sum"] = float(np.sum(tuning_total_sums)) if tuning_total_sums else None
            row["diagnostic_total_s_sum"] = float(np.sum(diagnostic_total_sums)) if diagnostic_total_sums else None
            row["n_total_params_mean"] = float(np.mean(total_params)) if total_params else None
            row["n_trainable_params_mean"] = float(np.mean(trainable_params)) if trainable_params else None
            rows.append(row)

        self._rank_rows(rows, within_key="within_model")
        rows.sort(key=lambda r: r["aggregate_rank_within_model"])
        best = rows[0] if rows else None
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sweep_name": f"{self._sweep_name}_{model['name']}",
            "model_name": model["name"],
            "model_kind": model["kind"],
            "n_configs": len(rows),
            "n_seeds": len(self._seeds),
            "enabled_tasks": list(task_summaries.keys()),
            "best_config_id": best["config_id"] if best else None,
            "best_overall": best,
            "task_summaries": task_summaries,
            "table": rows,
        }

    def _config_point_for(
        self,
        config_id: str,
        task_summaries: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        for summary in task_summaries.values():
            for cfg in summary["configs"]:
                if cfg["config_id"] == config_id:
                    return dict(cfg["config_point"])
        return {}

    def _add_task_columns(self, row: dict[str, Any], task_name: str, cfg: dict[str, Any]) -> None:
        rank_metric = self._ranking[task_name]["metric"]
        task_prefix = "mg" if task_name == "mackey_glass" else task_name
        metadata = cfg.get("metadata_mean", {})
        row[f"{task_prefix}_n_total_params"] = metadata.get("n_total_params")
        row[f"{task_prefix}_n_trainable_params"] = metadata.get("n_trainable_params")
        if task_name == "delay_recall":
            for metric in ("memory_corr_total", "memory_eff_total"):
                if metric in cfg.get("val_mean", {}):
                    row[f"delay_recall_{metric}_mean"] = cfg["val_mean"][metric]
                if metric in cfg.get("test_mean", {}):
                    row[f"delay_recall_test_{metric}_mean"] = cfg["test_mean"][metric]
            row["_rank_value_delay_recall"] = _scalar(cfg["val_mean"].get(rank_metric))
        elif task_name == "narma10":
            row["narma10_val_nmse_mean"] = cfg.get("val_mean", {}).get("nmse")
            row["narma10_test_nmse_mean"] = cfg.get("test_mean", {}).get("nmse")
            row["_rank_value_narma10"] = _scalar(cfg["val_mean"].get(rank_metric))
        elif task_name == "mackey_glass":
            row["mg_val_nmse_mean"] = cfg.get("val_mean", {}).get("nmse")
            row["mg_test_nmse_mean"] = cfg.get("test_mean", {}).get("nmse")
            row["_rank_value_mackey_glass"] = _scalar(cfg["val_mean"].get(rank_metric))

    def _rank_rows(self, rows: list[dict[str, Any]], within_key: str) -> None:
        enabled = self._enabled_tasks()
        rank_columns = []
        for task_name in enabled:
            direction = self._ranking[task_name]["direction"]
            raw_values = np.array([r.get(f"_rank_value_{task_name}", np.nan) for r in rows], dtype=float)
            if direction == "min":
                values = np.where(np.isfinite(raw_values), raw_values, np.inf)
                ranks = rankdata(values, method="min").astype(int)
            else:
                values = np.where(np.isfinite(raw_values), raw_values, -np.inf)
                ranks = rankdata(-values, method="min").astype(int)
            rank_col = self._rank_column(task_name, within_key)
            rank_columns.append(rank_col)
            for row, rank in zip(rows, ranks, strict=True):
                row[rank_col] = int(rank)
        for row in rows:
            row[f"aggregate_rank_{within_key}"] = float(np.mean([row[c] for c in rank_columns])) if rank_columns else 0.0

    def _rank_column(self, task_name: str, suffix: str) -> str:
        short = "mg" if task_name == "mackey_glass" else task_name
        if suffix == "global":
            return f"global_rank_{short}"
        return f"rank_{short}_{suffix}"

    def _build_comparison_summary(self, model_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for summary in model_summaries:
            for row in summary["table"]:
                rows.append(dict(row))
        if not rows:
            return []

        self._rank_rows(rows, within_key="global")
        baseline = self._baseline_row(rows)
        if baseline is not None:
            self._add_deltas(rows, baseline)
        rows.sort(key=lambda r: r["aggregate_rank_global"])
        for row in rows:
            row["global_aggregate_rank"] = row.pop("aggregate_rank_global")
        return rows

    def _baseline_row(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        baseline_rows = [r for r in rows if r.get("model_name") == "random_sparse"]
        if not baseline_rows:
            return None
        return min(baseline_rows, key=lambda r: r.get("aggregate_rank_global", float("inf")))

    def _add_deltas(self, rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
        for row in rows:
            if row is baseline or row.get("model_name") == baseline.get("model_name"):
                continue
            if "delay_recall" in self._enabled_tasks() and "delay_recall_memory_corr_total_mean" in row:
                row["delta_vs_baseline_delay_recall_memory_corr_total"] = (
                    row["delay_recall_memory_corr_total_mean"]
                    - baseline.get("delay_recall_memory_corr_total_mean", 0.0)
                )
            if "narma10" in self._enabled_tasks() and row.get("narma10_val_nmse_mean") is not None:
                row["delta_vs_baseline_narma10_val_nmse"] = (
                    row["narma10_val_nmse_mean"]
                    - baseline.get("narma10_val_nmse_mean", 0.0)
                )
            if "mackey_glass" in self._enabled_tasks() and row.get("mg_val_nmse_mean") is not None:
                row["delta_vs_baseline_mg_val_nmse"] = (
                    row["mg_val_nmse_mean"]
                    - baseline.get("mg_val_nmse_mean", 0.0)
                )

    def _save_run(self, run: dict[str, Any], runs_dir: Path) -> Path:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{run['config_id']}_seed{run['seed']}.json"
        _write_json(path, run)
        return path

    def _save_task_summary(self, summary: dict[str, Any], task_dir: Path) -> None:
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json(task_dir / "summary.json", summary)
        _write_csv(task_dir / "summary.csv", self._task_summary_rows(summary))

    def _task_summary_rows(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for cfg in summary["configs"]:
            row = {
                "config_id": cfg["config_id"],
                "n_seeds": cfg["n_seeds"],
                "best_config_id": summary["best_config_id"],
            }
            row.update(cfg["config_point"])
            _flatten_metric_block(row, "val", cfg.get("val_mean", {}), cfg.get("val_std", {}))
            _flatten_metric_block(row, "test", cfg.get("test_mean", {}), cfg.get("test_std", {}))
            row.update(_selected_time_fields(cfg))
            row["tuning_total_s_sum"] = summary.get("tuning_total_s_sum")
            row["diagnostic_total_s_sum"] = summary.get("diagnostic_total_s_sum")
            _flatten_prefixed(row, "timing_mean", cfg.get("timing_mean", {}))
            _flatten_prefixed(row, "timing_std", cfg.get("timing_std", {}))
            _flatten_prefixed(row, "timing_sum", cfg.get("timing_sum", {}))
            _flatten_prefixed(row, "metadata_mean", cfg.get("metadata_mean", {}))
            rows.append(row)
        return rows

    def _save_model_summary(self, summary: dict[str, Any], model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        rows = [_public_row(row) for row in summary["table"]]
        json_summary = dict(summary)
        json_summary["table"] = rows
        json_summary["best_overall"] = _public_row(summary["best_overall"])
        _write_json(model_dir / "summary.json", json_summary)
        _write_csv(model_dir / "summary.csv", rows)

    def _save_comparison_summary(
        self,
        comparison_table: list[dict[str, Any]],
        model_summaries: list[dict[str, Any]],
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        public_rows = [_public_row(row) for row in comparison_table]
        _write_csv(self._output_dir / "comparison_summary.csv", public_rows)

        best_by_model: dict[str, dict[str, Any]] = {}
        for row in public_rows:
            model_name = row["model_name"]
            current = best_by_model.get(model_name)
            if current is None or row.get("aggregate_rank_within_model", float("inf")) < current.get("aggregate_rank_within_model", float("inf")):
                best_by_model[model_name] = row

        counts = {summary["model_name"]: summary["n_configs"] for summary in model_summaries}
        unique_counts = set(counts.values())
        n_configs_per_model: Any = unique_counts.pop() if len(unique_counts) == 1 else counts
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_models": len(model_summaries),
            "n_configs_per_model": n_configs_per_model,
            "enabled_tasks": self._enabled_tasks(),
            "best_overall": public_rows[0] if public_rows else None,
            "best_by_model": best_by_model,
            "table": public_rows,
        }
        _write_json(self._output_dir / "comparison_summary.json", payload)
        try:
            from rc_lab.analysis.task_rankings import save_task_rankings
            save_task_rankings(payload, self._output_dir, top_n=20)
        except Exception as exc:  # pragma: no cover - post-hoc analysis must not fail runs
            print(f"task_rankings post-processing skipped: {exc}")


def _dedupe(values: list[str]) -> list[str]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _summary_config_by_id(summary: dict[str, Any], config_id: str) -> dict[str, Any]:
    for cfg in summary["configs"]:
        if cfg["config_id"] == config_id:
            return cfg
    raise KeyError(config_id)


def _metric_keys(group: list[dict[str, Any]], block: str, preferred: list[str]) -> list[str]:
    keys: list[str] = []
    for key in preferred:
        if any(key in run.get(block, {}) for run in group):
            keys.append(key)
    for run in group:
        for key in run.get(block, {}):
            if key not in keys:
                keys.append(key)
    return keys


def _has_complete_test_metrics(run: dict[str, Any], metric_names: list[str]) -> bool:
    test_metrics = run.get("test_metrics", {})
    return bool(test_metrics) and all(metric in test_metrics for metric in metric_names)


def _mean(values: list[Any]) -> Any:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    arr = np.asarray(valid, dtype=float)
    mean = np.mean(arr, axis=0)
    return float(mean) if np.ndim(mean) == 0 else mean.tolist()


def _std(values: list[Any]) -> Any:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    arr = np.asarray(valid, dtype=float)
    std = np.std(arr, axis=0, ddof=0)
    return float(std) if np.ndim(std) == 0 else std.tolist()


def _scalar(value: Any) -> float:
    if value is None:
        return float("nan")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(np.sum(arr))


def _aggregate_numeric_dict(
    group: list[dict[str, Any]],
    block: str,
    reducer: Any,
) -> dict[str, float]:
    keys = sorted({
        key
        for run in group
        for key, value in run.get(block, {}).items()
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))
    })
    out: dict[str, float] = {}
    for key in keys:
        values = [
            float(run[block][key])
            for run in group
            if _is_finite_number(run.get(block, {}).get(key))
        ]
        if values:
            out[key] = float(reducer(values))
    return out


def _selected_time_fields(cfg: dict[str, Any] | None) -> dict[str, float | None]:
    if cfg is None:
        return {
            "selected_fit_s_mean": None,
            "selected_fit_s_std": None,
            "selected_test_s_mean": None,
            "selected_test_s_std": None,
            "selected_total_s_mean": None,
            "selected_total_s_std": None,
        }
    timing_mean = cfg.get("timing_mean", {})
    timing_std = cfg.get("timing_std", {})
    fit_mean = _first_timing(timing_mean, ["fit_s"])
    fit_std = _first_timing(timing_std, ["fit_s"])
    test_mean = _first_timing(timing_mean, ["test_s", "final_test_s"])
    test_std = _first_timing(timing_std, ["test_s", "final_test_s"])
    total_mean, total_std = _combine_fit_test(
        fit_mean,
        fit_std,
        test_mean,
        test_std,
        _first_timing(timing_mean, ["selected_total_s"]),
        _first_timing(timing_std, ["selected_total_s"]),
    )
    if total_mean is None:
        total_mean = _first_timing(timing_mean, ["total_s"])
        total_std = _first_timing(timing_std, ["total_s"])
    return {
        "selected_fit_s_mean": fit_mean,
        "selected_fit_s_std": fit_std,
        "selected_test_s_mean": test_mean,
        "selected_test_s_std": test_std,
        "selected_total_s_mean": total_mean,
        "selected_total_s_std": total_std,
    }


def _first_timing(timing: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = timing.get(key)
        if _is_finite_number(value):
            return float(value)
    return None


def _combine_fit_test(
    fit_mean: float | None,
    fit_std: float | None,
    test_mean: float | None,
    test_std: float | None,
    fallback_mean: float | None,
    fallback_std: float | None,
) -> tuple[float | None, float | None]:
    if _is_finite_number(fit_mean) and _is_finite_number(test_mean):
        total_mean = float(fit_mean) + float(test_mean)
        variances = [
            float(value) ** 2
            for value in (fit_std, test_std)
            if _is_finite_number(value)
        ]
        total_std = float(np.sqrt(np.sum(variances))) if variances else None
        return total_mean, total_std
    mean = float(fallback_mean) if _is_finite_number(fallback_mean) else None
    std = float(fallback_std) if _is_finite_number(fallback_std) else None
    return mean, std


def _sum_run_timing(runs: list[dict[str, Any]], key: str) -> float | None:
    values = [_timing_from_run(run, key) for run in runs]
    finite = [float(value) for value in values if _is_finite_number(value)]
    return float(np.sum(finite)) if finite else None


def _timing_from_run(run: dict[str, Any], key: str) -> float | None:
    value = run.get("timing", {}).get(key)
    if value is None and key == "tuning_s" and run.get("evaluation_phase") == "validation":
        value = run.get("timing", {}).get("total_s")
    return float(value) if _is_finite_number(value) else None


def _add_optional(left: float | None, right: float | None) -> float | None:
    values = [float(value) for value in (left, right) if _is_finite_number(value)]
    return float(np.sum(values)) if values else None


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))


def _flatten_metric_block(
    row: dict[str, Any],
    prefix: str,
    means: dict[str, Any],
    stds: dict[str, Any],
) -> None:
    for key, value in means.items():
        row[f"{prefix}_{key}_mean"] = value
        row[f"{prefix}_{key}_std"] = stds.get(key)


def _flatten_prefixed(row: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def _public_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        key: value
        for key, value in row.items()
        if key != "config_point" and not key.startswith("_")
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = make_json_safe(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    safe = make_json_safe(value)
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, ensure_ascii=False, allow_nan=False)
    return "" if safe is None else safe
