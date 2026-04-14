from datetime import datetime, timezone
from typing import Any

import numpy as np

from rc_lab.runners.sweep_runner import ConfigSummary, SweepRunResult, SweepSummary


def aggregate_sweep_results(
    results: list[SweepRunResult],
    sweep_name: str,
    task_name: str,
    primary_metric: str = "nmse",
) -> SweepSummary:
    """
    Agrupa resultados por config_id, calcula media y std sobre seeds.

    La mejor configuración se identifica por val_mean[primary_metric] mínimo.
    El test queda reservado para evaluación final y reporte, no para selección.
    """
    # Agrupar por config_id
    groups: dict[str, list[SweepRunResult]] = {}
    for r in results:
        groups.setdefault(r.config_id, []).append(r)

    configs: list[ConfigSummary] = []
    for config_id, runs in groups.items():
        n_seeds = len(runs)
        config_point = runs[0].config_point

        # Recopilar métricas por split
        val_keys = list(runs[0].val_metrics.keys())
        test_keys = list(runs[0].test_metrics.keys())

        val_mean = {
            k: float(np.mean([r.val_metrics[k] for r in runs])) for k in val_keys
        }
        val_std = {
            k: float(np.std([r.val_metrics[k] for r in runs], ddof=0)) for k in val_keys
        }
        test_mean = {
            k: float(np.mean([r.test_metrics[k] for r in runs])) for k in test_keys
        }
        test_std = {
            k: float(np.std([r.test_metrics[k] for r in runs], ddof=0)) for k in test_keys
        }

        # Moda del best_ridge (valor más frecuente entre seeds)
        ridge_values = [r.best_ridge for r in runs]
        unique, counts = np.unique(ridge_values, return_counts=True)
        best_ridge_mode = float(unique[np.argmax(counts)])

        configs.append(ConfigSummary(
            config_id=config_id,
            config_point=config_point,
            n_seeds=n_seeds,
            val_mean=val_mean,
            val_std=val_std,
            test_mean=test_mean,
            test_std=test_std,
            best_ridge_mode=best_ridge_mode,
        ))

    # Mejor config por val_mean[primary_metric] — nunca por test
    best_config_id = min(
        configs,
        key=lambda c: c.val_mean.get(primary_metric, float("inf")),
    ).config_id

    seeds = sorted({r.seed for r in results})

    return SweepSummary(
        sweep_name=sweep_name,
        n_configs=len(configs),
        n_seeds=len(seeds),
        task_name=task_name,
        configs=configs,
        best_config_id=best_config_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def results_to_dataframe(results: list[SweepRunResult]) -> dict[str, list[Any]]:
    """
    Convierte la lista de SweepRunResult a un dict de listas compatible
    con csv.DictWriter, sin depender de pandas.

    Cada entrada del dict es una columna; cada posición es una fila.
    """
    if not results:
        return {}

    rows: dict[str, list[Any]] = {
        "config_id": [],
        "seed": [],
        "best_ridge": [],
    }

    # Columnas de config_point (inferidas del primer resultado)
    cp_keys = list(results[0].config_point.keys())
    for k in cp_keys:
        rows[k] = []

    # Columnas de métricas
    val_keys = list(results[0].val_metrics.keys())
    test_keys = list(results[0].test_metrics.keys())
    for k in val_keys:
        rows[f"val_{k}"] = []
    for k in test_keys:
        rows[f"test_{k}"] = []

    for r in results:
        rows["config_id"].append(r.config_id)
        rows["seed"].append(r.seed)
        rows["best_ridge"].append(r.best_ridge)
        for k in cp_keys:
            rows[k].append(r.config_point.get(k, None))
        for k in val_keys:
            rows[f"val_{k}"].append(r.val_metrics.get(k, None))
        for k in test_keys:
            rows[f"test_{k}"].append(r.test_metrics.get(k, None))

    return rows
