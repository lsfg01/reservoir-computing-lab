import dataclasses
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rc_lab.metrics.error import nmse, rmse
from rc_lab.models.esn import ESNModel
from rc_lab.readouts.ridge import RidgeReadout, build_readout_features
from rc_lab.readouts.ridge_sweep import RidgeParamSelector, RidgeSelectionResult
from rc_lab.reservoirs.diagnostics import reservoir_diagnostics as _reservoir_diagnostics
from rc_lab.runners.runner import resolve_reservoir
from rc_lab.tasks.base import BaseTask, TaskData
from rc_lab.utils.seeding import set_seed
from rc_lab.utils.timing import timer

ReadoutMode = Literal["states", "extended"]

_METRIC_FNS = {"nmse": nmse, "rmse": rmse}


# ---------------------------------------------------------------------------
# Dataclasses de resultados
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SweepRunResult:
    sweep_name: str
    config_id: str
    seed: int
    config_point: dict[str, Any]
    best_ridge: float
    val_curve: dict[str, float]   # keys son str para serialización JSON
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    timing: dict[str, float]
    timestamp: str
    reservoir_diagnostics: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ConfigSummary:
    config_id: str
    config_point: dict[str, Any]
    n_seeds: int
    val_mean: dict[str, float]
    val_std: dict[str, float]
    test_mean: dict[str, float]
    test_std: dict[str, float]
    best_ridge_mode: float


@dataclasses.dataclass
class SweepSummary:
    sweep_name: str
    n_configs: int
    n_seeds: int
    task_name: str
    configs: list[ConfigSummary]
    best_config_id: str   # config con menor val_mean[primary_metric]
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config_id(config_point: dict[str, Any]) -> str:
    """Hash determinista del config_point (ordenado por clave)."""
    canonical = json.dumps(config_point, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _resolve_task(name: str, state_policy: str = "reset", task_cfg: dict | None = None) -> BaseTask:
    from rc_lab.tasks.narma10 import Narma10Task
    from rc_lab.tasks.mackey_glass import MackeyGlassTask

    if task_cfg is None:
        task_cfg = {}

    if name == "narma10":
        return Narma10Task(state_policy=state_policy)
    elif name == "mackey_glass":
        return MackeyGlassTask(
            tau=task_cfg.get("tau", 17),
            dt=task_cfg.get("dt", 0.1),
            beta=task_cfg.get("beta", 0.2),
            gamma=task_cfg.get("gamma", 0.1),
            n=task_cfg.get("n", 10),
            prediction_horizon=task_cfg.get("prediction_horizon", 1),
            sample_stride=task_cfg.get("sample_stride", 1),
            discard_transient=task_cfg.get("discard_transient", 0),
            initial_history=task_cfg.get("initial_history", "constant"),
            history_value=task_cfg.get("history_value", 0.9),
            history_low=task_cfg.get("history_low", 1.1),
            history_high=task_cfg.get("history_high", 1.3),
            state_policy=state_policy,
        )
    else:
        raise ValueError(f"Tarea desconocida: {name!r}. Disponibles: ['narma10', 'mackey_glass']")


# ---------------------------------------------------------------------------
# SweepRunner
# ---------------------------------------------------------------------------

class SweepRunner:
    """
    Orquesta un barrido sistemático de hiperparámetros del baseline ESN.

    Para cada config_point × seed:
      1. Genera datos (train / val / test)
      2. Computa estados del reservoir
      3. Selecciona ridge_param por validación (RidgeParamSelector)
      4. Entrena readout final y evalúa en test
      5. Persiste SweepRunResult individual

    Al finalizar agrega resultados y persiste SweepSummary.
    """

    def __init__(self, sweep_config: dict[str, Any]) -> None:
        self._cfg = sweep_config
        self._sweep_name: str = sweep_config["sweep"]["name"]
        self._output_dir = Path(sweep_config["sweep"]["output_dir"])
        self._seeds: list[int] = sweep_config["sweep"]["seeds"]

        task_cfg = sweep_config["task"]
        self._task = _resolve_task(
            task_cfg["name"],
            state_policy=task_cfg.get("state_policy", "reset"),
            task_cfg=task_cfg,
        )
        self._task_name: str = task_cfg["name"]
        self._n_train: int = task_cfg["n_train"]
        self._n_val: int = task_cfg["n_val"]
        self._n_test: int = task_cfg["n_test"]
        self._washout: int = task_cfg["washout"]

        self._res_cfg: dict[str, Any] = sweep_config["reservoir"]
        self._N: int = self._res_cfg["N"]

        readout_cfg = sweep_config["readout"]
        self._ridge_candidates: list[float] = readout_cfg["ridge_candidates"]
        self._readout_mode: ReadoutMode = readout_cfg.get("features", "states")

        self._metric_names: list[str] = sweep_config.get("metrics", ["nmse"])
        self._primary_metric: str = self._task.primary_metric
        self._transient_kmax: int = sweep_config.get("diagnostics", {}).get("transient_kmax", 50)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def run(self) -> SweepSummary:
        """Ejecuta el sweep completo y devuelve el SweepSummary."""
        config_points = self._expand_grid()
        all_results: list[SweepRunResult] = []

        for config_point in config_points:
            config_id = make_config_id(config_point)
            for seed in self._seeds:
                set_seed(seed)
                task_data = self._task.generate(
                    n_train=self._n_train,
                    n_val=self._n_val,
                    n_test=self._n_test,
                    washout=self._washout,
                    seed=seed,
                )
                result = self._run_single(config_point, config_id, seed, task_data)
                self._save_run_result(result)
                all_results.append(result)

        from rc_lab.utils.aggregation import aggregate_sweep_results
        summary = aggregate_sweep_results(
            all_results,
            sweep_name=self._sweep_name,
            task_name=self._task_name,
            primary_metric=self._primary_metric,
        )
        self._save_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _expand_grid(self) -> list[dict[str, Any]]:
        """Producto cartesiano del grid de hiperparámetros."""
        grid: dict[str, list] = self._cfg["grid"]
        keys = list(grid.keys())
        values = list(grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _run_single(
        self,
        config_point: dict[str, Any],
        config_id: str,
        seed: int,
        task_data: TaskData,
    ) -> SweepRunResult:
        timing: dict[str, float] = {}

        with timer() as t_total:
            # Construir reservoir dinámicamente según el tipo configurado.
            # spectral_radius e input_scaling vienen del config_point (grid);
            # el resto de parámetros (type, N, sparsity, bias_scaling, etc.)
            # vienen del bloque reservoir del YAML.
            res_params = {
                **self._res_cfg,
                "spectral_radius": config_point["spectral_radius"],
                "input_scaling": config_point["input_scaling"],
            }
            # leak_rate pertenece a ESNModel, no al builder de matrices.
            res_params.pop("leak_rate", None)

            reservoir_builder = resolve_reservoir(res_params)
            n_inputs = task_data.u_train.shape[1]
            matrices = reservoir_builder.build(N=self._N, n_inputs=n_inputs, seed=seed)
            diag = _reservoir_diagnostics(matrices.W, transient_kmax=self._transient_kmax)

            leak_rate = config_point.get("leak_rate", 1.0)
            esn = ESNModel(matrices.W, matrices.Win, matrices.bias, leak_rate=leak_rate)

            # Estados train (washout incluido en u_train)
            with timer() as t_train:
                X_train, x_end_train = esn.run_states(
                    task_data.u_train, washout=self._washout
                )
            timing["train_states_s"] = t_train["elapsed"]

            # Y_train puntuado: descartar los primeros washout pasos
            u_train_post = task_data.u_train[self._washout:]
            Y_train = task_data.y_train[self._washout:]

            # Estados val y test — bifurcación por semántica de TaskData
            assert task_data.u_val is not None and task_data.y_val is not None

            if task_data.u_val_full is not None:
                # reset: cada split tiene su propio warmup de longitud washout
                X_val, _ = esn.run_states(
                    task_data.u_val_full, washout=self._washout, x0=None
                )
                X_test, _ = esn.run_states(
                    task_data.u_test_full, washout=self._washout, x0=None
                )
            else:
                # carryover: estado continuo desde train
                X_val, x_end_val = esn.run_states(
                    task_data.u_val, washout=0, x0=x_end_train
                )
                X_test, _ = esn.run_states(
                    task_data.u_test, washout=0, x0=x_end_val
                )

            # u_val / y_val / u_test / y_test ya son puntuados en TaskData
            u_val_post = task_data.u_val
            Y_val = task_data.y_val
            u_test_post = task_data.u_test
            Y_test = task_data.y_test

            # Design matrices
            F_train = build_readout_features(X_train, u_train_post, self._readout_mode)
            F_val = build_readout_features(X_val, u_val_post, self._readout_mode)
            F_test = build_readout_features(X_test, u_test_post, self._readout_mode)

            # Selección de ridge_param por validación
            with timer() as t_ridge:
                selector = RidgeParamSelector(self._ridge_candidates)
                ridge_result: RidgeSelectionResult = selector.select(
                    F_train, Y_train, F_val, Y_val
                )
            timing["ridge_select_s"] = t_ridge["elapsed"]

            # Readout final con best_ridge
            readout = RidgeReadout(ridge_param=ridge_result.best_ridge)
            readout.fit(F_train, Y_train)
            y_pred_val = readout.predict(F_val)
            y_pred_test = readout.predict(F_test)

            # Métricas
            val_metrics = {
                m: _METRIC_FNS[m](Y_val, y_pred_val)
                for m in self._metric_names
                if m in _METRIC_FNS
            }
            test_metrics = {
                m: _METRIC_FNS[m](Y_test, y_pred_test)
                for m in self._metric_names
                if m in _METRIC_FNS
            }

        timing["total_s"] = t_total["elapsed"]

        # val_curve con keys str para serialización JSON
        val_curve_str = {str(k): v for k, v in ridge_result.val_curve.items()}

        return SweepRunResult(
            sweep_name=self._sweep_name,
            config_id=config_id,
            seed=seed,
            config_point=config_point,
            best_ridge=ridge_result.best_ridge,
            val_curve=val_curve_str,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            timing=timing,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reservoir_diagnostics=diag,
        )

    def _save_run_result(self, result: SweepRunResult) -> None:
        from rc_lab.utils.io import save_sweep_run_result
        save_sweep_run_result(result, self._output_dir)

    def _save_summary(self, summary: SweepSummary) -> None:
        from rc_lab.utils.io import save_sweep_summary
        save_sweep_summary(summary, self._output_dir)
