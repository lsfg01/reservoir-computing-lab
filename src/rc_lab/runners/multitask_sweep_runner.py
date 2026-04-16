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
    """
    config_id: str
    config_point: dict[str, Any]
    n_seeds: int
    # NARMA-10: métrica primaria val y test
    narma10_primary_metric: str   # e.g. "nmse" o "rmse"
    narma10_val_mean: float
    narma10_val_std: float
    narma10_test_mean: float
    narma10_test_std: float
    # Mackey-Glass: métrica primaria val y test
    mg_primary_metric: str        # e.g. "nmse" o "rmse"
    mg_val_mean: float
    mg_val_std: float
    mg_test_mean: float
    mg_test_std: float
    # Memory Capacity
    mc_total_mean: float
    mc_total_std: float
    # Ranks por tarea (1 = mejor, según métrica primaria configurable)
    rank_narma10: int
    rank_mg: int
    rank_mc: int
    # Agregación inter-tarea
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
    """
    task_params: dict[str, Any] = dict(mt_config["tasks"][task_name])

    return {
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
    ) -> None:
        self._ranking_config = ranking_config
        self._shortlist_top_n = shortlist_top_n

    def aggregate(
        self,
        narma10_summary: Any,   # SweepSummary
        mg_summary: Any,        # SweepSummary
        mc_summary: MCTaskSummary,
    ) -> "MultiTaskSweepSummary":
        import numpy as np

        # --- Construir índices por config_id ---
        n10_by_id = {c.config_id: c for c in narma10_summary.configs}
        mg_by_id  = {c.config_id: c for c in mg_summary.configs}
        mc_by_id  = {c.config_id: c for c in mc_summary.configs}

        # --- Verificar alineación ---
        ids_n10 = set(n10_by_id)
        ids_mg  = set(mg_by_id)
        ids_mc  = set(mc_by_id)
        if ids_n10 != ids_mg or ids_n10 != ids_mc:
            missing_mg  = ids_n10 - ids_mg
            missing_mc  = ids_n10 - ids_mc
            extra_mg    = ids_mg  - ids_n10
            extra_mc    = ids_mc  - ids_n10
            raise ValueError(
                f"config_id mismatch between tasks. "
                f"Missing in mackey_glass: {missing_mg}, extra: {extra_mg}. "
                f"Missing in mc: {missing_mc}, extra: {extra_mc}."
            )

        config_ids = sorted(n10_by_id.keys())
        n = len(config_ids)

        # --- Verificar base estadística uniforme entre tareas ---
        asymmetric = []
        for cid in config_ids:
            ns_n10 = n10_by_id[cid].n_seeds
            ns_mg  = mg_by_id[cid].n_seeds
            ns_mc  = mc_by_id[cid].n_seeds
            if not (ns_n10 == ns_mg == ns_mc):
                asymmetric.append(
                    f"{cid}: narma10={ns_n10}, mackey_glass={ns_mg}, mc={ns_mc}"
                )
        if asymmetric:
            raise ValueError(
                "Base estadística asimétrica entre tareas — no se puede construir el ranking. "
                "Configs afectadas:\n" + "\n".join(asymmetric)
            )

        # --- Extraer métricas primarias ---
        n10_spec = self._ranking_config["narma10"]
        mg_spec  = self._ranking_config["mackey_glass"]
        mc_spec  = self._ranking_config["memory_capacity"]

        primary_n10 = np.array([n10_by_id[cid].val_mean[n10_spec.metric] for cid in config_ids])
        primary_mg  = np.array([mg_by_id[cid].val_mean[mg_spec.metric]   for cid in config_ids])
        primary_mc  = np.array([mc_by_id[cid].mc_total_mean               for cid in config_ids])

        # --- Calcular ranks (1 = mejor) ---
        ranks_n10 = self._rank(primary_n10, n10_spec.direction)
        ranks_mg  = self._rank(primary_mg,  mg_spec.direction)
        ranks_mc  = self._rank(primary_mc,  mc_spec.direction)

        # --- aggregate_rank = media aritmética de los tres ranks ---
        agg_ranks = (ranks_n10 + ranks_mg + ranks_mc) / 3.0

        # --- Construir entradas ---
        entries: list[MultiTaskConfigEntry] = []
        for i, cid in enumerate(config_ids):
            n10c = n10_by_id[cid]
            mgc  = mg_by_id[cid]
            mcc  = mc_by_id[cid]
            entries.append(MultiTaskConfigEntry(
                config_id=cid,
                config_point=n10c.config_point,
                n_seeds=n10c.n_seeds,
                narma10_primary_metric=n10_spec.metric,
                narma10_val_mean=n10c.val_mean[n10_spec.metric],
                narma10_val_std=n10c.val_std[n10_spec.metric],
                narma10_test_mean=n10c.test_mean[n10_spec.metric],
                narma10_test_std=n10c.test_std[n10_spec.metric],
                mg_primary_metric=mg_spec.metric,
                mg_val_mean=mgc.val_mean[mg_spec.metric],
                mg_val_std=mgc.val_std[mg_spec.metric],
                mg_test_mean=mgc.test_mean[mg_spec.metric],
                mg_test_std=mgc.test_std[mg_spec.metric],
                mc_total_mean=mcc.mc_total_mean,
                mc_total_std=mcc.mc_total_std,
                rank_narma10=int(ranks_n10[i]),
                rank_mg=int(ranks_mg[i]),
                rank_mc=int(ranks_mc[i]),
                aggregate_rank=float(agg_ranks[i]),
            ))

        # --- Ordenar por aggregate_rank ascendente ---
        entries.sort(key=lambda e: e.aggregate_rank)

        # --- Shortlist: top-N ---
        top_n = min(self._shortlist_top_n, n)
        shortlist = [e.config_id for e in entries[:top_n]]

        return MultiTaskSweepSummary(
            sweep_name=narma10_summary.sweep_name.replace("_narma10", ""),
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

        # --- Fase 1: NARMA-10 via SweepRunner ---
        from rc_lab.runners.sweep_runner import SweepRunner
        narma10_cfg = build_subtask_config(self._cfg, "narma10")
        narma10_summary = SweepRunner(narma10_cfg).run()

        # --- Fase 2: Mackey-Glass via SweepRunner ---
        mg_cfg = build_subtask_config(self._cfg, "mackey_glass")
        mg_summary = SweepRunner(mg_cfg).run()

        # --- Fase 3: Memory Capacity ---
        mc_runs = self._run_mc_phase()

        # --- Fase 4: Agregación y persistencia ---
        mc_summary = aggregate_mc_results(mc_runs, self._sweep_name)
        aggregator = MultiTaskAggregator(
            ranking_config=self._ranking_config,
            shortlist_top_n=self._shortlist_top_n,
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
        from rc_lab.reservoirs.random_sparse import RandomSparseReservoir
        from rc_lab.runners.sweep_runner import make_config_id
        from rc_lab.utils.timing import timer
        from rc_lab.utils.io import save_mc_run_result

        mc_dir = self._output_dir / "mc"
        config_points = self._expand_grid()
        mc_runs: list[MCRunResult] = []

        for config_point in config_points:
            config_id = make_config_id(config_point)
            for seed in self._seeds:
                # No se llama a set_seed: el reservoir se construye con seed explícita
                # y MemoryCapacityEvaluator usa RNG local con seed — reproducibilidad garantizada
                reservoir = RandomSparseReservoir(
                    spectral_radius=config_point["spectral_radius"],
                    input_scaling=config_point["input_scaling"],
                    sparsity=self._res_cfg.get("sparsity", 0.9),
                    leak_rate=config_point.get("leak_rate", 1.0),
                    bias_scaling=self._res_cfg.get("bias_scaling", 0.0),
                )
                N = self._res_cfg["N"]
                matrices = reservoir.build(N=N, n_inputs=1, seed=seed)
                esn = ESNModel(
                    matrices.W, matrices.Win, matrices.bias,
                    leak_rate=config_point.get("leak_rate", 1.0),
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
