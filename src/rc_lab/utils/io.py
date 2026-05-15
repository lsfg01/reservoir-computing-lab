import dataclasses
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def make_json_safe(obj: Any) -> Any:
    """
    Recursively converts obj to a JSON-safe structure:
    - dataclasses → dict (via dataclasses.asdict)
    - dict → dict with values recursively sanitised
    - list/tuple → list with elements recursively sanitised
    - non-finite float (inf, -inf, nan) → None
    - all other values → unchanged
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return make_json_safe(obj.tolist())
        if isinstance(obj, np.generic):
            return make_json_safe(obj.item())
    except ImportError:
        pass
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    """Carga un archivo YAML y devuelve el dict resultante."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_result(result: Any, output_dir: str | Path) -> Path:
    """
    Guarda un RunResult como JSON en output_dir/{experiment}_seed{seed}.json.

    Devuelve el Path del archivo creado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{result.experiment}_seed{result.seed}.json"
    path = output_dir / filename

    # Serializar dataclass a dict
    data = dataclasses.asdict(result)
    # Añadir timestamp si no está presente
    if "timestamp" not in data or not data["timestamp"]:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
    data = make_json_safe(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)

    return path


# ---------------------------------------------------------------------------
# Persistencia del sweep
# ---------------------------------------------------------------------------

def save_sweep_run_result(result: Any, output_dir: str | Path) -> Path:
    """
    Guarda un SweepRunResult como JSON en:
        {output_dir}/runs/{config_id}_seed{seed}.json

    El JSON incluye el config_point completo para ser auto-contenido.
    """
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{result.config_id}_seed{result.seed}.json"
    path = runs_dir / filename

    data = dataclasses.asdict(result)
    data = make_json_safe(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)

    return path


def save_sweep_summary(summary: Any, output_dir: str | Path) -> tuple[Path, Path]:
    """
    Guarda SweepSummary en dos formatos:
        {output_dir}/summary.json  — estructura completa
        {output_dir}/summary.csv   — tabla plana (una fila por config_id)

    Devuelve (json_path, csv_path).
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON completo
    json_path = output_dir / "summary.json"
    data = dataclasses.asdict(summary)
    data = make_json_safe(data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)

    # CSV plano
    csv_path = output_dir / "summary.csv"
    grid_keys: list[str] = []
    metric_keys: list[str] = []
    for cfg in summary.configs:
        for key in cfg.config_point:
            if key not in grid_keys:
                grid_keys.append(key)
        for key in cfg.val_mean:
            if key not in metric_keys:
                metric_keys.append(key)
        for key in cfg.test_mean:
            if key not in metric_keys:
                metric_keys.append(key)

    fieldnames = ["config_id"] + grid_keys + ["best_ridge_mode"]
    for metric in metric_keys:
        fieldnames.extend([
            f"val_{metric}_mean",
            f"val_{metric}_std",
            f"test_{metric}_mean",
            f"test_{metric}_std",
        ])
    fieldnames.append("n_seeds")

    def _csv_value(value: Any) -> Any:
        safe = make_json_safe(value)
        if isinstance(safe, (list, dict)):
            return json.dumps(safe, ensure_ascii=False, allow_nan=False)
        return safe if safe is not None else ""

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for cfg in summary.configs:
            row = {
                "config_id": cfg.config_id,
                "best_ridge_mode": cfg.best_ridge_mode,
                "n_seeds": cfg.n_seeds,
            }
            for key in grid_keys:
                row[key] = _csv_value(cfg.config_point.get(key, ""))
            for metric in metric_keys:
                row[f"val_{metric}_mean"] = _csv_value(cfg.val_mean.get(metric, ""))
                row[f"val_{metric}_std"] = _csv_value(cfg.val_std.get(metric, ""))
                row[f"test_{metric}_mean"] = _csv_value(cfg.test_mean.get(metric, ""))
                row[f"test_{metric}_std"] = _csv_value(cfg.test_std.get(metric, ""))
            writer.writerow(row)

    return json_path, csv_path


def load_sweep_results(output_dir: str | Path) -> list[Any]:
    """
    Carga todos los JSON de {output_dir}/runs/ y reconstruye la lista
    de SweepRunResult. Útil para re-agregar sin re-ejecutar el sweep.
    """
    from rc_lab.runners.sweep_runner import SweepRunResult

    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        return []

    results = []
    for json_file in sorted(runs_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Compatibilidad hacia atrás: ficheros JSON anteriores a esta feature
        # no contienen el campo reservoir_diagnostics.
        data.setdefault("reservoir_diagnostics", {})
        results.append(SweepRunResult(**data))

    return results


# ---------------------------------------------------------------------------
# Persistencia del barrido multi-tarea
# ---------------------------------------------------------------------------

def save_mc_run_result(result: Any, output_dir: str | Path) -> Path:
    """
    Guarda un MCRunResult como JSON en:
        {output_dir}/{config_id}_seed{seed}.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{result.config_id}_seed{result.seed}.json"
    path = output_dir / filename

    data = dataclasses.asdict(result)
    data = make_json_safe(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)

    return path


def save_multitask_summary(summary: Any, output_dir: str | Path) -> tuple[Path, Path, Path]:
    """
    Guarda MultiTaskSweepSummary en tres artefactos:
        {output_dir}/summary.json     — estructura completa con ranking_config
        {output_dir}/summary.csv      — tabla plana (una fila por config)
        {output_dir}/shortlist.json   — config_ids de la shortlist con métricas

    Las columnas del CSV para métricas predictivas usan nombres planos explícitos
    derivados de la métrica primaria configurada por tarea, e.g.:
        narma10_val_nmse_mean, narma10_test_nmse_mean, mg_val_nmse_mean, ...

    Devuelve (json_path, csv_path, shortlist_path).
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- summary.json ---
    json_path = output_dir / "summary.json"
    data = dataclasses.asdict(summary)
    data = make_json_safe(data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)

    # --- summary.csv ---
    # Determinar qué tareas están habilitadas (backward-compat: si no existe el
    # atributo, asumir las tres activas para preservar el comportamiento anterior)
    enabled_tasks = getattr(summary, "enabled_tasks", ["narma10", "mackey_glass", "memory_capacity"])
    mg_enabled  = "mackey_glass"    in enabled_tasks
    n10_enabled = "narma10"         in enabled_tasks
    mc_enabled  = "memory_capacity" in enabled_tasks

    # Determinar métricas primarias sólo para las tareas habilitadas
    if summary.configs:
        n10_metric = summary.configs[0].narma10_primary_metric if n10_enabled else None
        mg_metric  = summary.configs[0].mg_primary_metric      if mg_enabled  else None
    else:
        n10_metric = "nmse" if n10_enabled else None
        mg_metric  = "nmse" if mg_enabled  else None

    # Columnas del grid (orden estable)
    grid_keys = list(summary.grid.keys()) if summary.grid else []

    # Construir fieldnames condicionalmente según tareas habilitadas
    fieldnames = ["config_id"] + grid_keys
    if n10_enabled:
        fieldnames += [
            f"narma10_val_{n10_metric}_mean",
            f"narma10_val_{n10_metric}_std",
            f"narma10_test_{n10_metric}_mean",
            f"narma10_test_{n10_metric}_std",
            "rank_narma10",
        ]
    if mg_enabled:
        fieldnames += [
            f"mg_val_{mg_metric}_mean",
            f"mg_val_{mg_metric}_std",
            f"mg_test_{mg_metric}_mean",
            f"mg_test_{mg_metric}_std",
            "rank_mg",
        ]
    if mc_enabled:
        fieldnames += ["mc_total_mean", "mc_total_std", "rank_mc"]
    fieldnames += ["aggregate_rank", "n_seeds"]

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in summary.configs:
            row: dict[str, Any] = {"config_id": entry.config_id}
            for k in grid_keys:
                row[k] = entry.config_point.get(k, "")
            if n10_enabled:
                row[f"narma10_val_{n10_metric}_mean"]  = entry.narma10_val_mean
                row[f"narma10_val_{n10_metric}_std"]   = entry.narma10_val_std
                row[f"narma10_test_{n10_metric}_mean"] = entry.narma10_test_mean
                row[f"narma10_test_{n10_metric}_std"]  = entry.narma10_test_std
                row["rank_narma10"] = entry.rank_narma10
            if mg_enabled:
                row[f"mg_val_{mg_metric}_mean"]  = entry.mg_val_mean
                row[f"mg_val_{mg_metric}_std"]   = entry.mg_val_std
                row[f"mg_test_{mg_metric}_mean"] = entry.mg_test_mean
                row[f"mg_test_{mg_metric}_std"]  = entry.mg_test_std
                row["rank_mg"] = entry.rank_mg
            if mc_enabled:
                row["mc_total_mean"] = entry.mc_total_mean
                row["mc_total_std"]  = entry.mc_total_std
                row["rank_mc"]       = entry.rank_mc
            row["aggregate_rank"] = entry.aggregate_rank
            row["n_seeds"]        = entry.n_seeds
            writer.writerow(row)

    # --- shortlist.json ---
    shortlist_ids = set(summary.shortlist)
    shortlist_entries = [
        dataclasses.asdict(e) for e in summary.configs
        if e.config_id in shortlist_ids
    ]
    shortlist_entries = make_json_safe(shortlist_entries)
    shortlist_path = output_dir / "shortlist.json"
    with open(shortlist_path, "w", encoding="utf-8") as f:
        json.dump(
            {"shortlist_top_n": summary.shortlist_top_n, "configs": shortlist_entries},
            f, indent=2, allow_nan=False,
        )

    return json_path, csv_path, shortlist_path
