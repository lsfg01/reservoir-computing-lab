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
        results.append(SweepRunResult(**data))

    return results
