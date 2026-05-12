import dataclasses
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

from rc_lab.metrics.error import nmse, rmse
from rc_lab.models.esn import ESNModel
from rc_lab.readouts.ridge import RidgeReadout, build_readout_features
from rc_lab.reservoirs.base import BaseReservoirBuilder
from rc_lab.tasks.base import BaseTask
from rc_lab.utils.io import save_result
from rc_lab.utils.seeding import set_seed
from rc_lab.utils.timing import timer

StatePolicy = Literal["reset", "carryover"]
ReadoutMode = Literal["states", "extended"]

# ---------------------------------------------------------------------------
# Resolvers: mapeo nombre/tipo → clase concreta
# ---------------------------------------------------------------------------

def resolve_task(name: str, state_policy: str = "reset", task_cfg: dict | None = None) -> BaseTask:
    """Instancia la tarea correspondiente al nombre dado."""
    from rc_lab.tasks.narma10 import Narma10Task

    if task_cfg is None:
        task_cfg = {}

    if name == "narma10":
        return Narma10Task(state_policy=state_policy)

    if name == "mackey_glass":
        from rc_lab.tasks.mackey_glass import MackeyGlassTask
        return MackeyGlassTask(
            tau=task_cfg.get("tau", 17),
            dt=task_cfg.get("dt", 0.1),
            state_policy=state_policy,
        )

    raise ValueError(f"Tarea desconocida: {name!r}. Disponibles: ['narma10', 'mackey_glass']")


def resolve_reservoir(reservoir_cfg: dict[str, Any]) -> BaseReservoirBuilder:
    """Instancia el reservoir builder a partir del bloque de configuración."""
    from rc_lab.reservoirs.random_sparse import RandomSparseReservoir
    from rc_lab.reservoirs.cycle import CycleReservoir
    from rc_lab.reservoirs.cycle_jump import CycleJumpReservoir
    from rc_lab.reservoirs.nonnormal_chain import NonnormalChainReservoir

    registry: dict[str, type[BaseReservoirBuilder]] = {
        "random_sparse":   RandomSparseReservoir,
        "cycle":           CycleReservoir,
        "cycle_jump":      CycleJumpReservoir,
        "nonnormal_chain": NonnormalChainReservoir,
    }

    rtype = reservoir_cfg.get("type", "random_sparse")
    if rtype not in registry:
        available = list(registry.keys())
        raise ValueError(
            f"Tipo de reservoir desconocido: {rtype!r}. "
            f"Disponibles: {available}"
        )

    # Extraer sólo los parámetros del constructor (excluir 'type' y 'N')
    params = {k: v for k, v in reservoir_cfg.items() if k not in ("type", "N")}
    return registry[rtype](**params)


# ---------------------------------------------------------------------------
# RunResult (clase del resultado de la run)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunResult:
    experiment: str
    seed: int
    config: dict[str, Any]
    metrics: dict[str, float]
    timing: dict[str, float]
    timestamp: str


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

_METRIC_FNS = {
    "nmse": nmse,
    "rmse": rmse,
}


class ExperimentRunner:
    """
    Orquesta una corrida completa del experimento.
    Delega en Task, Reservoir, ESN, Readout y Metrics.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        for key in ("experiment", "task", "reservoir", "readout", "metrics"):
            if key not in config:
                raise ValueError(f"Falta la clave requerida en config: {key!r}")

        self._config = config
        exp = config["experiment"] 
        self._state_policy: StatePolicy = exp.get("state_policy", "reset") #eliminar ya que en la sweep-upgrade state_policy cambia a la tarea
        self._readout_mode: ReadoutMode = exp.get("readout_features", "states")

        if self._state_policy not in ("reset", "carryover"):
            raise ValueError(f"state_policy inválido: {self._state_policy!r}")
        if self._readout_mode not in ("states", "extended"):
            raise ValueError(f"readout_features inválido: {self._readout_mode!r}")

    def run(self, seed: int) -> RunResult:
        """Ejecuta una corrida completa para una semilla dada."""
        cfg = self._config
        exp_cfg = cfg["experiment"]
        task_cfg = cfg["task"]
        res_cfg = cfg["reservoir"]
        readout_cfg = cfg["readout"]
        metric_names: list[str] = cfg["metrics"]

        timing: dict[str, float] = {}

        with timer() as t_total:
            # 1. Fijar semilla
            set_seed(seed)

            # 2. Generar datos de la tarea
            task = resolve_task(
                task_cfg["name"],
                state_policy=task_cfg.get("state_policy", "reset"),
                task_cfg=task_cfg,
            )
            task_data = task.generate(
                n_train=task_cfg["n_train"],
                n_val=0,
                n_test=task_cfg["n_test"],
                washout=task_cfg["washout"],
                seed=seed,
            )
            washout = task_data.washout

            # 3. Construir reservoir
            N = res_cfg["N"]
            n_inputs = task_data.u_train.shape[1]
            reservoir = resolve_reservoir(res_cfg)
            matrices = reservoir.build(N=N, n_inputs=n_inputs, seed=seed)

            # 4. Instanciar ESNModel
            leak_rate = res_cfg.get("leak_rate", 1.0)
            esn = ESNModel(matrices.W, matrices.Win, matrices.bias, leak_rate=leak_rate)

            # 5. Fase de entrenamiento
            with timer() as t_train:
                X_train, x_final = esn.run_states(task_data.u_train, washout=washout)
                u_train_post = task_data.u_train[washout:]
                Y_train = task_data.y_train[washout:]
                F_train = build_readout_features(X_train, u_train_post, self._readout_mode)
                readout = RidgeReadout(ridge_param=readout_cfg.get("ridge_param", 1e-6))
                readout.fit(F_train, Y_train)
            timing["train_s"] = t_train["elapsed"]

            # 6. Fase de test — semántica delegada en TaskData
            with timer() as t_test:
                if task_data.u_test_full is not None:
                    # reset: la task preparó u_test_full con warmup explícito
                    X_test, _ = esn.run_states(
                        task_data.u_test_full, washout=washout, x0=None
                    )
                else:
                    # carryover: estado continuo desde train
                    X_test, _ = esn.run_states(
                        task_data.u_test, washout=0, x0=x_final
                    )
                # u_test y y_test son siempre los arrays puntuados
                F_test = build_readout_features(X_test, task_data.u_test, self._readout_mode)
                y_pred = readout.predict(F_test)
            timing["test_s"] = t_test["elapsed"]

            # 7. Calcular métricas
            y_test = task_data.y_test
            metrics: dict[str, float] = {}
            for m in metric_names:
                if m not in _METRIC_FNS:
                    raise ValueError(f"Métrica desconocida: {m!r}")
                metrics[m] = _METRIC_FNS[m](y_test, y_pred)

        timing["total_s"] = t_total["elapsed"]

        result = RunResult(
            experiment=exp_cfg["name"],
            seed=seed,
            config=cfg,
            metrics=metrics,
            timing=timing,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # 8. Persistir resultado
        output_dir = exp_cfg.get("output_dir", "results")
        save_result(result, output_dir)

        return result

    def run_multi_seed(self, seeds: list[int]) -> list[RunResult]:
        """Ejecuta run(seed) para cada semilla y devuelve la lista de resultados."""
        return [self.run(seed) for seed in seeds]
