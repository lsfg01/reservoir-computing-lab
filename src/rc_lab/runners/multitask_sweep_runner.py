"""
multitask_sweep_runner.py

Orquestador del barrido multi-tarea sobre la infraestructura ESN existente.
Evalúa cada configuración del reservoir en NARMA-10, Mackey-Glass y Memory
Capacity, agrega los resultados por seeds, y produce un ranking inter-tarea.

Este módulo define los dataclasses de resultados y contendrá en tareas
posteriores la lógica de agregación, ranking y ejecución del sweep.
"""

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rc_lab.reservoirs.diagnostics import reservoir_diagnostics as _reservoir_diagnostics


# ---------------------------------------------------------------------------
# Configuración de ranking
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RankingSpec:
    """Especificación de ranking para una tarea: métrica primaria y dirección."""
    metric: str     # nombre de la métrica primaria ("nmse", "rmse", "mc_total")
    direction: str  # "min" o "max"


# ---------------------------------------------------------------------------
# Dataclasses de resultados de Memory Capacity
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MCRunResult:
    """Resultado de una evaluación MC para una configuración y seed concretas."""
    sweep_name: str
    config_id: str
    seed: int               # identifica la realización completa: construcción del
                            # reservoir y generación de la señal de entrada de MC
    config_point: dict[str, Any]
    mc_total: float
    kmax: int
    timing: dict[str, float]
    timestamp: str
    reservoir_diagnostics: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class MCConfigResult:
    """Resultados MC agregados (media y std) para una configuración sobre todas las seeds."""
    config_id: str
    config_point: dict[str, Any]
    n_seeds: int
    mc_total_mean: float
    mc_total_std: float


@dataclasses.dataclass
class MCTaskSummary:
    """Contenedor de resultados MC agregados para todas las configuraciones del barrido."""
    sweep_name: str
    n_configs: int
    n_seeds: int
    configs: list[MCConfigResult]
    timestamp: str


# ---------------------------------------------------------------------------
# Dataclasses del resultado multi-tarea
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MultiTaskConfigEntry:
    """
    Entrada de resultado multi-tarea para una configuración.

    Estructura estable e independiente del YAML. Los campos {task}_primary_metric
    almacenan el nombre de la métrica elegida (e.g. "nmse") para que la capa de
    exportación pueda construir nombres de columna explícitos en el CSV
    (e.g. narma10_val_nmse_mean).

    Los campos de tareas deshabilitadas se establecen a None. Los campos de rank
    de tareas deshabilitadas se establecen a None también.
    """
    config_id: str
    config_point: dict[str, Any]
    n_seeds: int
    # NARMA-10: métrica primaria val y test (None si la tarea está deshabilitada)
    narma10_primary_metric: str | None   # e.g. "nmse" o "rmse"
    narma10_val_mean: float | None
    narma10_val_std: float | None
    narma10_test_mean: float | None
    narma10_test_std: float | None
    # Mackey-Glass: métrica primaria val y test (None si la tarea está deshabilitada)
    mg_primary_metric: str | None        # e.g. "nmse" o "rmse"
    mg_val_mean: float | None
    mg_val_std: float | None
    mg_test_mean: float | None
    mg_test_std: float | None
    # Memory Capacity (None si la tarea está deshabilitada)
    mc_total_mean: float | None
    mc_total_std: float | None
    # Ranks por tarea (1 = mejor, según métrica primaria configurable; None si deshabilitada)
    rank_narma10: int | None
    rank_mg: int | None
    rank_mc: int | None
    # Agregación inter-tarea (media de ranks de tareas activas únicamente)
    aggregate_rank: float


@dataclasses.dataclass
class MultiTaskSweepSummary:
    """Resultado final del barrido multi-tarea."""
    sweep_name: str
    n_configs: int
    n_seeds: int
    grid: dict[str, list]
    ranking_config: dict[str, RankingSpec]   # métrica primaria y dirección por tarea
    configs: list[MultiTaskConfigEntry]       # ordenados por aggregate_rank asc
    shortlist: list[str]                      # config_ids de la shortlist
    shortlist_top_n: int
    timestamp: str
    enabled_tasks: list[str] = dataclasses.field(default_factory=lambda: ["narma10", "mackey_glass", "memory_capacity"])


# ---------------------------------------------------------------------------
# aggregate_mc_results
# ---------------------------------------------------------------------------

def aggregate_mc_results(mc_runs: list[MCRunResult], sweep_name: str) -> MCTaskSummary:
    """
    Agrega una lista de MCRunResult por config_id.

    Preconditions:
    - mc_runs es no vacía
    - todos los elementos tienen config_id y mc_total válidos

    Postconditions:
    - mc_total_mean = media aritmética de mc_total sobre seeds
    - mc_total_std  = desviación estándar (ddof=0) sobre seeds
    - n_seeds       = número de seeds por config
    """
    import numpy as np

    # Agrupar por config_id preservando el orden de primera aparición
    groups: dict[str, list[MCRunResult]] = {}
    for run in mc_runs:
        groups.setdefault(run.config_id, []).append(run)

    configs: list[MCConfigResult] = []
    for config_id, runs in groups.items():
        values = np.array([r.mc_total for r in runs], dtype=float)
        configs.append(MCConfigResult(
            config_id=config_id,
            config_point=runs[0].config_point,
            n_seeds=len(runs),
            mc_total_mean=float(np.mean(values)),
            mc_total_std=float(np.std(values, ddof=0)),
        ))

    return MCTaskSummary(
        sweep_name=sweep_name,
        n_configs=len(configs),
        n_seeds=len(next(iter(groups.values()))),
        configs=configs,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# build_subtask_config
# ---------------------------------------------------------------------------

def build_subtask_config(mt_config: dict[str, Any], task_name: str) -> dict[str, Any]:
    """
    Construye un dict de configuración compatible con SweepRunner para una
    tarea predictiva (narma10 o mackey_glass) a partir de la config multi-tarea.

    Postconditions:
    - El dict resultante contiene las claves: sweep, task, reservoir, grid, readout, metrics
    - sweep.name       = {mt_config.sweep.name}_{task_name}
    - sweep.output_dir = {mt_config.sweep.output_dir}/{task_name}
    - sweep.seeds      = mt_config.sweep.seeds (sin modificación)
    - grid y readout se propagan sin modificación
    - diagnostics se propaga si está presente en mt_config
    """
    task_params: dict[str, Any] = dict(mt_config["tasks"][task_name])

    subtask_cfg: dict[str, Any] = {
        "sweep": {
            "name": f"{mt_config['sweep']['name']}_{task_name}",
            "output_dir": f"{mt_config['sweep']['output_dir']}/{task_name}",
            "seeds": mt_config["sweep"]["seeds"],
        },
        "task": {
            "name": task_name,
            **task_params,
        },
        "reservoir": mt_config["reservoir"],
        "grid": mt_config["grid"],
        "readout": mt_config["readout"],
        "metrics": mt_config["metrics"],
    }

    # Propagar bloque diagnostics si está presente en la config raíz
    if "diagnostics" in mt_config:
        subtask_cfg["diagnostics"] = mt_config["diagnostics"]

    return subtask_cfg


# ---------------------------------------------------------------------------
# MultiTaskAggregator
# ---------------------------------------------------------------------------

class MultiTaskAggregator:
    """
    Recibe los summaries por tarea y produce el MultiTaskSweepSummary con
    ranks por tarea, aggregate_rank y shortlist.

    El ranking es configurable por tarea: direction="min" → rank 1 = menor valor,
    direction="max" → rank 1 = mayor valor.
    """

    def __init__(
        self,
        ranking_config: dict[str, "RankingSpec"],
        shortlist_top_n: int = 10,
        enabled_tasks: "list[str] | None" = None,
    ) -> None:
        self._ranking_config = ranking_config
        self._shortlist_top_n = shortlist_top_n
        # Si no se especifica, usar las tres tareas (comportamiento original)
        self._enabled_tasks: list[str] = (
            enabled_tasks
            if enabled_tasks is not None
            else ["narma10", "mackey_glass", "memory_capacity"]
        )

    def aggregate(
        self,
        narma10_summary: Any,   # SweepSummary or None
        mg_summary: Any,        # SweepSummary or None
        mc_summary: "MCTaskSummary | None",
    ) -> "MultiTaskSweepSummary":
        import numpy as np

        enabled = self._enabled_tasks

        # --- Construir índices por config_id para tareas activas ---
        n10_by_id = {c.config_id: c for c in narma10_summary.configs} if narma10_summary is not None else {}
        mg_by_id  = {c.config_id: c for c in mg_summary.configs}      if mg_summary is not None else {}
        mc_by_id  = {c.config_id: c for c in mc_summary.configs}      if mc_summary is not None else {}

        # Determinar el conjunto de config_ids desde las tareas activas
        active_id_sets: list[set[str]] = []
        if "narma10" in enabled and n10_by_id:
            active_id_sets.append(set(n10_by_id))
        if "mackey_glass" in enabled and mg_by_id:
            active_id_sets.append(set(mg_by_id))
        if "memory_capacity" in enabled and mc_by_id:
            active_id_sets.append(set(mc_by_id))

        if not active_id_sets:
            raise ValueError("No hay tareas activas para agregar.")

        # Verificar alineación entre tareas activas
        reference_ids = active_id_sets[0]
        for id_set in active_id_sets[1:]:
            if reference_ids != id_set:
                raise ValueError(
                    f"config_id mismatch entre tareas activas: "
                    f"{reference_ids.symmetric_difference(id_set)}"
                )

        config_ids = sorted(reference_ids)
        n = len(config_ids)

        # --- Verificar base estadística uniforme entre tareas activas ---
        asymmetric = []
        for cid in config_ids:
            seed_counts = {}
            if "narma10" in enabled and n10_by_id:
                seed_counts["narma10"] = n10_by_id[cid].n_seeds
            if "mackey_glass" in enabled and mg_by_id:
                seed_counts["mackey_glass"] = mg_by_id[cid].n_seeds
            if "memory_capacity" in enabled and mc_by_id:
                seed_counts["memory_capacity"] = mc_by_id[cid].n_seeds
            if len(set(seed_counts.values())) > 1:
                asymmetric.append(f"{cid}: " + ", ".join(f"{k}={v}" for k, v in seed_counts.items()))
        if asymmetric:
            raise ValueError(
                "Base estadística asimétrica entre tareas — no se puede construir el ranking. "
                "Configs afectadas:\n" + "\n".join(asymmetric)
            )

        # --- Extraer métricas primarias y calcular ranks para tareas activas ---
        rank_arrays: dict[str, "np.ndarray"] = {}

        if "narma10" in enabled and n10_by_id:
            n10_spec = self._ranking_config["narma10"]
            primary_n10 = np.array([n10_by_id[cid].val_mean[n10_spec.metric] for cid in config_ids])
            rank_arrays["narma10"] = self._rank(primary_n10, n10_spec.direction)

        if "mackey_glass" in enabled and mg_by_id:
            mg_spec = self._ranking_config["mackey_glass"]
            primary_mg = np.array([mg_by_id[cid].val_mean[mg_spec.metric] for cid in config_ids])
            rank_arrays["mackey_glass"] = self._rank(primary_mg, mg_spec.direction)

        if "memory_capacity" in enabled and mc_by_id:
            mc_spec = self._ranking_config["memory_capacity"]
            primary_mc = np.array([mc_by_id[cid].mc_total_mean for cid in config_ids])
            rank_arrays["memory_capacity"] = self._rank(primary_mc, mc_spec.direction)

        # --- aggregate_rank = media aritmética de los ranks de tareas activas ---
        if rank_arrays:
            agg_ranks = np.mean(list(rank_arrays.values()), axis=0)
        else:
            agg_ranks = np.zeros(n)

        # --- Construir entradas ---
        # Obtener n_seeds desde la primera tarea activa disponible
        def _n_seeds_for(cid: str) -> int:
            if "narma10" in enabled and n10_by_id:
                return n10_by_id[cid].n_seeds
            if "mackey_glass" in enabled and mg_by_id:
                return mg_by_id[cid].n_seeds
            if "memory_capacity" in enabled and mc_by_id:
                return mc_by_id[cid].n_seeds
            return 0

        def _config_point_for(cid: str) -> dict:
            if "narma10" in enabled and n10_by_id:
                return n10_by_id[cid].config_point
            if "mackey_glass" in enabled and mg_by_id:
                return mg_by_id[cid].config_point
            if "memory_capacity" in enabled and mc_by_id:
                return mc_by_id[cid].config_point
            return {}

        _nan = float("nan")
        entries: list[MultiTaskConfigEntry] = []
        for i, cid in enumerate(config_ids):
            n10_spec = self._ranking_config.get("narma10")
            mg_spec  = self._ranking_config.get("mackey_glass")

            # narma10 fields
            if "narma10" in enabled and n10_by_id and n10_spec:
                n10c = n10_by_id[cid]
                narma10_primary_metric = n10_spec.metric
                narma10_val_mean  = n10c.val_mean[n10_spec.metric]
                narma10_val_std   = n10c.val_std[n10_spec.metric]
                narma10_test_mean = n10c.test_mean[n10_spec.metric]
                narma10_test_std  = n10c.test_std[n10_spec.metric]
                rank_narma10      = int(rank_arrays["narma10"][i])
            else:
                narma10_primary_metric = None
                narma10_val_mean = narma10_val_std = narma10_test_mean = narma10_test_std = None
                rank_narma10 = None

            # mackey_glass fields
            if "mackey_glass" in enabled and mg_by_id and mg_spec:
                mgc = mg_by_id[cid]
                mg_primary_metric = mg_spec.metric
                mg_val_mean  = mgc.val_mean[mg_spec.metric]
                mg_val_std   = mgc.val_std[mg_spec.metric]
                mg_test_mean = mgc.test_mean[mg_spec.metric]
                mg_test_std  = mgc.test_std[mg_spec.metric]
                rank_mg      = int(rank_arrays["mackey_glass"][i])
            else:
                mg_primary_metric = None
                mg_val_mean = mg_val_std = mg_test_mean = mg_test_std = None
                rank_mg = None

            # memory_capacity fields
            if "memory_capacity" in enabled and mc_by_id:
                mcc = mc_by_id[cid]
                mc_total_mean = mcc.mc_total_mean
                mc_total_std  = mcc.mc_total_std
                rank_mc       = int(rank_arrays["memory_capacity"][i])
            else:
                mc_total_mean = mc_total_std = None
                rank_mc = None

            entries.append(MultiTaskConfigEntry(
                config_id=cid,
                config_point=_config_point_for(cid),
                n_seeds=_n_seeds_for(cid),
                narma10_primary_metric=narma10_primary_metric,
                narma10_val_mean=narma10_val_mean,
                narma10_val_std=narma10_val_std,
                narma10_test_mean=narma10_test_mean,
                narma10_test_std=narma10_test_std,
                mg_primary_metric=mg_primary_metric,
                mg_val_mean=mg_val_mean,
                mg_val_std=mg_val_std,
                mg_test_mean=mg_test_mean,
                mg_test_std=mg_test_std,
                mc_total_mean=mc_total_mean,
                mc_total_std=mc_total_std,
                rank_narma10=rank_narma10,
                rank_mg=rank_mg,
                rank_mc=rank_mc,
                aggregate_rank=float(agg_ranks[i]),
            ))

        # --- Ordenar por aggregate_rank ascendente ---
        entries.sort(key=lambda e: e.aggregate_rank)

        # --- Shortlist: top-N ---
        top_n = min(self._shortlist_top_n, n)
        shortlist = [e.config_id for e in entries[:top_n]]

        # Determinar sweep_name desde el primer summary activo disponible
        if narma10_summary is not None:
            base_sweep_name = narma10_summary.sweep_name.replace("_narma10", "")
        elif mg_summary is not None:
            base_sweep_name = mg_summary.sweep_name.replace("_mackey_glass", "")
        elif mc_summary is not None:
            base_sweep_name = mc_summary.sweep_name
        else:
            base_sweep_name = ""

        return MultiTaskSweepSummary(
            sweep_name=base_sweep_name,
            n_configs=n,
            n_seeds=entries[0].n_seeds if entries else 0,
            grid={},   # poblado por el runner
            ranking_config=self._ranking_config,
            configs=entries,
            shortlist=shortlist,
            shortlist_top_n=top_n,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _rank(values: "Any", direction: str) -> "Any":
        """
        Calcula ranks enteros (1 = mejor) para un array de valores.
        direction="min" → rank 1 = menor valor
        direction="max" → rank 1 = mayor valor
        Desempate: orden estable por posición (method="ordinal").
        """
        import numpy as np
        from scipy.stats import rankdata

        if direction == "min":
            return rankdata(values, method="ordinal").astype(int)
        elif direction == "max":
            return rankdata([-v for v in values], method="ordinal").astype(int)
        else:
            raise ValueError(f"direction debe ser 'min' o 'max', recibido: {direction!r}")


# ---------------------------------------------------------------------------
# MultiTaskSweepRunner
# ---------------------------------------------------------------------------

class MultiTaskSweepRunner:
    """
    Orquesta el barrido multi-tarea completo:
      Fase 1 — NARMA-10 via SweepRunner
      Fase 2 — Mackey-Glass via SweepRunner
      Fase 3 — Memory Capacity via MemoryCapacityEvaluator
      Fase 4 — Agregación inter-tarea y persistencia

    No modifica SweepRunner ni MemoryCapacityEvaluator.
    """

    _REQUIRED_TASK_BLOCKS = ("narma10", "mackey_glass", "memory_capacity")
    _SUPPORTED_METRICS = ("nmse", "rmse")

    def __init__(self, sweep_config: dict[str, Any]) -> None:
        self._cfg = sweep_config
        self._validate_config(sweep_config)

        self._sweep_name: str = sweep_config["sweep"]["name"]
        self._output_dir = Path(sweep_config["sweep"]["output_dir"])
        self._seeds: list[int] = sweep_config["sweep"]["seeds"]
        self._grid: dict[str, list] = sweep_config["grid"]
        self._res_cfg: dict[str, Any] = sweep_config["reservoir"]
        self._metrics: list[str] = sweep_config["metrics"]

        # Construir MemoryCapacityEvaluator con los parámetros de tasks.memory_capacity
        mc_cfg = sweep_config["tasks"]["memory_capacity"]
        from rc_lab.evaluators.memory_capacity import MemoryCapacityEvaluator
        self._mc_evaluator = MemoryCapacityEvaluator(
            washout=mc_cfg.get("washout", 200),
            input_length=mc_cfg.get("input_length", 3000),
            fit_fraction=mc_cfg.get("fit_fraction", 0.5),
            kmax=mc_cfg.get("kmax", None),
            ridge_param=mc_cfg.get("ridge_param", 1e-6),
        )

        # Construir ranking_config desde sweep_config["ranking"]
        ranking_raw = sweep_config["ranking"]
        self._ranking_config: dict[str, RankingSpec] = {
            "narma10": RankingSpec(
                metric=ranking_raw["narma10"]["metric"],
                direction=ranking_raw["narma10"]["direction"],
            ),
            "mackey_glass": RankingSpec(
                metric=ranking_raw["mackey_glass"]["metric"],
                direction=ranking_raw["mackey_glass"]["direction"],
            ),
            "memory_capacity": RankingSpec(
                metric=ranking_raw["memory_capacity"]["metric"],
                direction=ranking_raw["memory_capacity"]["direction"],
            ),
        }
        self._shortlist_top_n: int = ranking_raw.get("shortlist_top_n", 10)

    # ------------------------------------------------------------------
    # Validación de config
    # ------------------------------------------------------------------

    def _validate_config(self, cfg: dict[str, Any]) -> None:
        """Valida la config multi-tarea antes de ejecutar ninguna corrida."""
        # Bloques de primer nivel requeridos
        for key in ("sweep", "reservoir", "grid", "tasks", "readout", "metrics", "ranking"):
            if key not in cfg:
                raise ValueError(f"Config multi-tarea: falta el bloque requerido '{key}'")

        # seeds
        if "seeds" not in cfg["sweep"] or not cfg["sweep"]["seeds"]:
            raise ValueError("Config multi-tarea: 'sweep.seeds' debe ser una lista no vacía")

        # Bloques de tarea requeridos
        tasks = cfg.get("tasks", {})
        for task in self._REQUIRED_TASK_BLOCKS:
            if task not in tasks:
                raise ValueError(
                    f"Config multi-tarea: falta el bloque requerido 'tasks.{task}'"
                )

        # Validación completa del bloque ranking
        metrics_calculated: list[str] = cfg.get("metrics", [])
        ranking = cfg.get("ranking", {})
        valid_directions = {"min", "max"}

        for task in ("narma10", "mackey_glass"):
            if task not in ranking:
                raise ValueError(
                    f"Config multi-tarea: falta 'ranking.{task}' en el bloque ranking"
                )
            task_rank = ranking[task] if isinstance(ranking[task], dict) else {}
            primary = task_rank.get("metric")
            direction = task_rank.get("direction")
            if primary is None:
                raise ValueError(f"Config multi-tarea: falta 'ranking.{task}.metric'")
            if primary not in metrics_calculated:
                raise ValueError(
                    f"Config multi-tarea: la métrica de ranking '{primary}' para '{task}' "
                    f"no está en metrics[] = {metrics_calculated}"
                )
            if direction not in valid_directions:
                raise ValueError(
                    f"Config multi-tarea: 'ranking.{task}.direction' debe ser 'min' o 'max', "
                    f"recibido: {direction!r}"
                )

        # memory_capacity: metric debe ser "mc_total", direction "min" o "max"
        if "memory_capacity" not in ranking:
            raise ValueError(
                "Config multi-tarea: falta 'ranking.memory_capacity' en el bloque ranking"
            )
        mc_rank = ranking["memory_capacity"] if isinstance(ranking["memory_capacity"], dict) else {}
        mc_metric = mc_rank.get("metric")
        mc_direction = mc_rank.get("direction")
        if mc_metric != "mc_total":
            raise ValueError(
                f"Config multi-tarea: 'ranking.memory_capacity.metric' debe ser 'mc_total', "
                f"recibido: {mc_metric!r}"
            )
        if mc_direction not in valid_directions:
            raise ValueError(
                f"Config multi-tarea: 'ranking.memory_capacity.direction' debe ser 'min' o 'max', "
                f"recibido: {mc_direction!r}"
            )

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self) -> MultiTaskSweepSummary:
        """Ejecuta el barrido multi-tarea completo y devuelve el summary."""
        from rc_lab.utils.timing import timer

        enabled_tasks: list[str] = []

        # --- Fase 1: NARMA-10 via SweepRunner ---
        from rc_lab.runners.sweep_runner import SweepRunner
        if self._cfg["tasks"]["narma10"].get("enabled", True):
            narma10_cfg = build_subtask_config(self._cfg, "narma10")
            narma10_summary = SweepRunner(narma10_cfg).run()
            enabled_tasks.append("narma10")
        else:
            narma10_summary = None

        # --- Fase 2: Mackey-Glass via SweepRunner ---
        if self._cfg["tasks"]["mackey_glass"].get("enabled", True):
            mg_cfg = build_subtask_config(self._cfg, "mackey_glass")
            mg_summary = SweepRunner(mg_cfg).run()
            enabled_tasks.append("mackey_glass")
        else:
            mg_summary = None

        # --- Fase 3: Memory Capacity ---
        if self._cfg["tasks"]["memory_capacity"].get("enabled", True):
            mc_runs = self._run_mc_phase()
            mc_summary = aggregate_mc_results(mc_runs, self._sweep_name)
            enabled_tasks.append("memory_capacity")
        else:
            mc_summary = None

        # --- Fase 4: Agregación y persistencia ---
        aggregator = MultiTaskAggregator(
            ranking_config=self._ranking_config,
            shortlist_top_n=self._shortlist_top_n,
            enabled_tasks=enabled_tasks,
        )
        mt_summary = aggregator.aggregate(narma10_summary, mg_summary, mc_summary)

        # Poblar grid en el summary (el aggregator lo deja vacío)
        mt_summary = MultiTaskSweepSummary(
            sweep_name=self._sweep_name,
            n_configs=mt_summary.n_configs,
            n_seeds=mt_summary.n_seeds,
            grid=self._grid,
            ranking_config=mt_summary.ranking_config,
            configs=mt_summary.configs,
            shortlist=mt_summary.shortlist,
            shortlist_top_n=mt_summary.shortlist_top_n,
            timestamp=mt_summary.timestamp,
            enabled_tasks=enabled_tasks,
        )

        from rc_lab.utils.io import save_multitask_summary
        save_multitask_summary(mt_summary, self._output_dir)

        return mt_summary

    # ------------------------------------------------------------------
    # Fase 3: MC
    # ------------------------------------------------------------------

    def _run_mc_phase(self) -> list[MCRunResult]:
        """Evalúa MC para cada config_point × seed. Propaga excepciones sin capturar."""
        import itertools
        from datetime import datetime, timezone

        from rc_lab.models.esn import ESNModel
        from rc_lab.runners.runner import resolve_reservoir
        from rc_lab.runners.sweep_runner import make_config_id
        from rc_lab.utils.timing import timer
        from rc_lab.utils.io import save_mc_run_result

        mc_dir = self._output_dir / "mc"
        config_points = self._expand_grid()
        mc_runs: list[MCRunResult] = []

        # Obtener transient_kmax desde el bloque diagnostics de la config
        transient_kmax: int = self._cfg.get("diagnostics", {}).get("transient_kmax", 50)

        for config_point in config_points:
            config_id = make_config_id(config_point)
            for seed in self._seeds:
                # No se llama a set_seed: el reservoir se construye con seed explícita
                # y MemoryCapacityEvaluator usa RNG local con seed — reproducibilidad garantizada

                # Construir parámetros del reservoir: bloque YAML + overrides del grid.
                # leak_rate pertenece a ESNModel, no al builder — se elimina antes de
                # llamar a resolve_reservoir.
                res_params = {
                    **self._res_cfg,
                    **{
                        k: v
                        for k, v in config_point.items()
                        if k not in ("leak_rate", "ridge_param")
                    },
                }
                res_params.pop("leak_rate", None)
                res_params.pop("ridge_param", None)

                reservoir_builder = resolve_reservoir(res_params)
                N = self._res_cfg["N"]
                matrices = reservoir_builder.build(N=N, n_inputs=1, seed=seed)
                diag = _reservoir_diagnostics(matrices.W, transient_kmax=transient_kmax)

                leak_rate = config_point.get("leak_rate", 1.0)
                esn = ESNModel(
                    matrices.W, matrices.Win, matrices.bias,
                    leak_rate=leak_rate,
                )

                with timer() as t:
                    mc_result = self._mc_evaluator.evaluate_details(esn, seed)

                run = MCRunResult(
                    sweep_name=self._sweep_name,
                    config_id=config_id,
                    seed=seed,
                    config_point=config_point,
                    mc_total=mc_result.mc_total,
                    kmax=mc_result.kmax,
                    timing={"mc_s": t["elapsed"]},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    reservoir_diagnostics=diag,
                )
                save_mc_run_result(run, mc_dir)
                mc_runs.append(run)

        return mc_runs

    def _expand_grid(self) -> list[dict[str, Any]]:
        """Producto cartesiano del grid de hiperparámetros."""
        import itertools
        keys = list(self._grid.keys())
        values = list(self._grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
