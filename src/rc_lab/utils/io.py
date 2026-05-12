import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


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

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    # CSV plano
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "config_id",
        "spectral_radius",
        "input_scaling",
        "leak_rate",
        "best_ridge_mode",
        "val_nmse_mean",
        "val_nmse_std",
        "test_nmse_mean",
        "test_nmse_std",
        "n_seeds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for cfg in summary.configs:
            row = {
                "config_id": cfg.config_id,
                "spectral_radius": cfg.config_point.get("spectral_radius", ""),
                "input_scaling": cfg.config_point.get("input_scaling", ""),
                "leak_rate": cfg.config_point.get("leak_rate", ""),
                "best_ridge_mode": cfg.best_ridge_mode,
                "val_nmse_mean": cfg.val_mean.get("nmse", ""),
                "val_nmse_std": cfg.val_std.get("nmse", ""),
                "test_nmse_mean": cfg.test_mean.get("nmse", ""),
                "test_nmse_std": cfg.test_std.get("nmse", ""),
                "n_seeds": cfg.n_seeds,
            }
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    # --- summary.csv ---
    # Determinar métricas primarias desde el primer entry (o desde ranking_config)
    if summary.configs:
        n10_metric = summary.configs[0].narma10_primary_metric
        mg_metric  = summary.configs[0].mg_primary_metric
    else:
        n10_metric = "nmse"
        mg_metric  = "nmse"

    # Columnas del grid (orden estable)
    grid_keys = list(summary.grid.keys()) if summary.grid else []

    fieldnames = (
        ["config_id"]
        + grid_keys
        + [
            f"narma10_val_{n10_metric}_mean",
            f"narma10_val_{n10_metric}_std",
            f"narma10_test_{n10_metric}_mean",
            f"narma10_test_{n10_metric}_std",
            f"mg_val_{mg_metric}_mean",
            f"mg_val_{mg_metric}_std",
            f"mg_test_{mg_metric}_mean",
            f"mg_test_{mg_metric}_std",
            "mc_total_mean",
            "mc_total_std",
            "rank_narma10",
            "rank_mg",
            "rank_mc",
            "aggregate_rank",
            "n_seeds",
        ]
    )

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in summary.configs:
            row: dict[str, Any] = {"config_id": entry.config_id}
            for k in grid_keys:
                row[k] = entry.config_point.get(k, "")
            row[f"narma10_val_{n10_metric}_mean"]  = entry.narma10_val_mean
            row[f"narma10_val_{n10_metric}_std"]   = entry.narma10_val_std
            row[f"narma10_test_{n10_metric}_mean"] = entry.narma10_test_mean
            row[f"narma10_test_{n10_metric}_std"]  = entry.narma10_test_std
            row[f"mg_val_{mg_metric}_mean"]        = entry.mg_val_mean
            row[f"mg_val_{mg_metric}_std"]         = entry.mg_val_std
            row[f"mg_test_{mg_metric}_mean"]       = entry.mg_test_mean
            row[f"mg_test_{mg_metric}_std"]        = entry.mg_test_std
            row["mc_total_mean"]   = entry.mc_total_mean
            row["mc_total_std"]    = entry.mc_total_std
            row["rank_narma10"]    = entry.rank_narma10
            row["rank_mg"]         = entry.rank_mg
            row["rank_mc"]         = entry.rank_mc
            row["aggregate_rank"]  = entry.aggregate_rank
            row["n_seeds"]         = entry.n_seeds
            writer.writerow(row)

    # --- shortlist.json ---
    shortlist_ids = set(summary.shortlist)
    shortlist_entries = [
        dataclasses.asdict(e) for e in summary.configs
        if e.config_id in shortlist_ids
    ]
    shortlist_path = output_dir / "shortlist.json"
    with open(shortlist_path, "w", encoding="utf-8") as f:
        json.dump(
            {"shortlist_top_n": summary.shortlist_top_n, "configs": shortlist_entries},
            f, indent=2, default=str,
        )

    return json_path, csv_path, shortlist_path
