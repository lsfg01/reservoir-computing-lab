from datetime import datetime, timezone
from typing import Any

import numpy as np

from rc_lab.runners.sweep_runner import ConfigSummary, SweepRunResult, SweepSummary


def aggregate_sweep_results(
    results: list[SweepRunResult],
    sweep_name: str,
    task_name: str,
    primary_metric: str = "nmse",
    primary_direction: str = "min",
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

        val_mean = {k: _mean_metric([r.val_metrics[k] for r in runs]) for k in val_keys}
        val_std = {k: _std_metric([r.val_metrics[k] for r in runs]) for k in val_keys}
        test_mean = {k: _mean_metric([r.test_metrics[k] for r in runs]) for k in test_keys}
        test_std = {k: _std_metric([r.test_metrics[k] for r in runs]) for k in test_keys}
        timing_mean = _aggregate_timing(runs, np.mean)
        timing_std = _aggregate_timing(runs, np.std)
        timing_sum = _aggregate_timing(runs, np.sum)

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
            timing_mean=timing_mean,
            timing_std=timing_std,
            timing_sum=timing_sum,
        ))

    # Mejor config por val_mean[primary_metric] — nunca por test
    if primary_direction == "min":
        best_config_id = min(
            configs,
            key=lambda c: _scalar_for_rank(c.val_mean.get(primary_metric, float("inf")), default=float("inf")),
        ).config_id
    elif primary_direction == "max":
        best_config_id = max(
            configs,
            key=lambda c: _scalar_for_rank(c.val_mean.get(primary_metric, float("-inf")), default=float("-inf")),
        ).config_id
    else:
        raise ValueError(f"primary_direction debe ser 'min' o 'max', recibido: {primary_direction!r}")

    seeds = sorted({r.seed for r in results})
    best_config = next(c for c in configs if c.config_id == best_config_id)
    selected_fit_s_mean = best_config.timing_mean.get("fit_s")
    selected_fit_s_std = best_config.timing_std.get("fit_s")
    selected_test_s_mean = _first_timing(best_config.timing_mean, ["test_s", "final_test_s"])
    selected_test_s_std = _first_timing(best_config.timing_std, ["test_s", "final_test_s"])
    selected_total_s_mean, selected_total_s_std = _combine_fit_test(
        selected_fit_s_mean,
        selected_fit_s_std,
        selected_test_s_mean,
        selected_test_s_std,
        best_config.timing_mean.get("selected_total_s"),
        best_config.timing_std.get("selected_total_s"),
    )
    tuning_total_s_sum = _sum_available(
        cfg.timing_sum.get("tuning_s")
        for cfg in configs
    )
    diagnostic_total_s_sum = _sum_available(
        cfg.timing_sum.get("total_s")
        for cfg in configs
    )

    return SweepSummary(
        sweep_name=sweep_name,
        n_configs=len(configs),
        n_seeds=len(seeds),
        task_name=task_name,
        configs=configs,
        best_config_id=best_config_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        selected_fit_s_mean=selected_fit_s_mean,
        selected_fit_s_std=selected_fit_s_std,
        selected_test_s_mean=selected_test_s_mean,
        selected_test_s_std=selected_test_s_std,
        selected_total_s_mean=selected_total_s_mean,
        selected_total_s_std=selected_total_s_std,
        tuning_total_s_sum=tuning_total_s_sum,
        diagnostic_total_s_sum=diagnostic_total_s_sum,
    )


def _mean_metric(values: list[Any]) -> Any:
    arr = np.asarray(values, dtype=float)
    mean = np.mean(arr, axis=0)
    return float(mean) if np.ndim(mean) == 0 else mean.tolist()


def _std_metric(values: list[Any]) -> Any:
    arr = np.asarray(values, dtype=float)
    std = np.std(arr, axis=0, ddof=0)
    return float(std) if np.ndim(std) == 0 else std.tolist()


def _scalar_for_rank(value: Any, default: float) -> float:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return default
    if arr.ndim == 0:
        return float(arr)
    return float(np.sum(arr))


def _aggregate_timing(
    runs: list[SweepRunResult],
    reducer: Any,
) -> dict[str, float]:
    keys = sorted({
        key
        for run in runs
        for key, value in run.timing.items()
        if _is_finite_number(value)
    })
    out: dict[str, float] = {}
    for key in keys:
        values = [
            float(run.timing[key])
            for run in runs
            if _is_finite_number(run.timing.get(key))
        ]
        if values:
            out[key] = float(reducer(values))
    return out


def _sum_available(values: Any) -> float | None:
    finite = [float(v) for v in values if _is_finite_number(v)]
    return float(np.sum(finite)) if finite else None


def _first_timing(timing: dict[str, float], keys: list[str]) -> float | None:
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


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))


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
