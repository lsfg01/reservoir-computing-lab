"""
design_comparison_runner.py

Orquestador de la comparación controlada entre familias de reservoir.

Recibe una configuración con una lista `designs`, itera sobre cada diseño,
construye una config compatible con `MultiTaskSweepRunner` (sustituyendo el
bloque `reservoir`), ejecuta el sweep, y recoge los `MultiTaskSweepSummary`.

La agregación comparativa (tabla, ranks globales, deltas vs baseline) se
implementa en `aggregate_comparison`.

La persistencia de la tabla comparativa se implementa en `save_comparison`.

Requisitos: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 8.4
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rc_lab.runners.multitask_sweep_runner import MultiTaskSweepRunner, MultiTaskSweepSummary


# ---------------------------------------------------------------------------
# build_design_config
# ---------------------------------------------------------------------------

def build_design_config(
    base_cfg: dict[str, Any],
    design: dict[str, Any],
) -> dict[str, Any]:
    """
    Produce un dict compatible con ``MultiTaskSweepRunner`` para un diseño
    concreto, sustituyendo el bloque ``reservoir`` y ajustando ``sweep.name``
    y ``sweep.output_dir``.

    Parameters
    ----------
    base_cfg:
        Configuración base del ``DesignComparisonRunner`` (contiene ``sweep``,
        ``designs``, ``grid``, ``tasks``, ``readout``, ``metrics``, ``ranking``).
    design:
        Entrada de la lista ``designs`` con al menos las claves ``name`` y
        ``reservoir``.

    Returns
    -------
    dict compatible con ``MultiTaskSweepRunner.__init__``:

    .. code-block:: python

        {
            "sweep": {
                "name": f"{base_name}_{design_name}",
                "output_dir": f"{base_output_dir}/{design_name}",
                "seeds": base_seeds,
            },
            "reservoir": design["reservoir"],
            "grid": base_grid,
            "tasks": base_tasks,
            "readout": base_readout,
            "metrics": base_metrics,
            "ranking": base_ranking,
        }

    Postconditions
    --------------
    - ``sweep.name``       == ``{base_name}_{design_name}``
    - ``sweep.output_dir`` == ``{base_output_dir}/{design_name}``
    - ``sweep.seeds``      == ``base_cfg["sweep"]["seeds"]`` (sin modificación)
    - ``reservoir``        == ``design["reservoir"]``
    - ``grid``, ``tasks``, ``readout``, ``metrics``, ``ranking`` se propagan
      sin modificación desde ``base_cfg``.
    """
    base_sweep = base_cfg["sweep"]
    base_name = base_sweep["name"]
    base_output_dir = base_sweep["output_dir"]
    base_seeds = base_sweep["seeds"]

    design_name: str = design["name"]

    return {
        "sweep": {
            "name": f"{base_name}_{design_name}",
            "output_dir": f"{base_output_dir}/{design_name}",
            "seeds": base_seeds,
        },
        "reservoir": design["reservoir"],
        "grid": base_cfg["grid"],
        "tasks": base_cfg["tasks"],
        "readout": base_cfg["readout"],
        "metrics": base_cfg["metrics"],
        "ranking": base_cfg["ranking"],
    }


# ---------------------------------------------------------------------------
# DesignComparisonRunner
# ---------------------------------------------------------------------------

class DesignComparisonRunner:
    """
    Orquesta la comparación controlada entre familias de reservoir.

    Para cada diseño de la lista ``designs``, construye una config compatible
    con ``MultiTaskSweepRunner``, ejecuta el sweep y recoge el
    ``MultiTaskSweepSummary``.

    La agregación comparativa (tabla, ranks globales, deltas vs baseline) se
    implementa en ``aggregate_comparison`` (tarea 8.2).

    Parameters
    ----------
    cfg:
        Configuración del experimento de comparación. Debe contener:
        - ``sweep``: ``name``, ``output_dir``, ``seeds``
        - ``designs``: lista de dicts con ``name`` y ``reservoir``
        - ``grid``, ``tasks``, ``readout``, ``metrics``, ``ranking``
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._validate_config(cfg)

        # Sweep base
        sweep = cfg["sweep"]
        self._sweep_name: str = sweep["name"]
        self._output_dir = Path(sweep["output_dir"])
        self._seeds: list[int] = sweep["seeds"]

        # Lista de diseños: cada uno tiene "name" y "reservoir"
        self._designs: list[dict[str, Any]] = cfg["designs"]

        # Bloques compartidos entre diseños
        self._grid: dict[str, list] = cfg["grid"]
        self._tasks: dict[str, Any] = cfg["tasks"]
        self._readout: dict[str, Any] = cfg["readout"]
        self._metrics: list[str] = cfg["metrics"]
        self._ranking: dict[str, Any] = cfg["ranking"]

    # ------------------------------------------------------------------
    # Validación de config
    # ------------------------------------------------------------------

    def _validate_config(self, cfg: dict[str, Any]) -> None:
        """Valida la config del DesignComparisonRunner antes de ejecutar."""
        required_keys = ("sweep", "designs", "grid", "tasks", "readout", "metrics", "ranking")
        for key in required_keys:
            if key not in cfg:
                raise ValueError(
                    f"Config de DesignComparisonRunner: falta el bloque requerido '{key}'"
                )

        # sweep
        sweep = cfg["sweep"]
        for sweep_key in ("name", "output_dir", "seeds"):
            if sweep_key not in sweep:
                raise ValueError(
                    f"Config de DesignComparisonRunner: falta 'sweep.{sweep_key}'"
                )
        if not sweep["seeds"]:
            raise ValueError(
                "Config de DesignComparisonRunner: 'sweep.seeds' debe ser una lista no vacía"
            )

        # designs
        designs = cfg["designs"]
        if not isinstance(designs, list) or len(designs) == 0:
            raise ValueError(
                "Config de DesignComparisonRunner: 'designs' debe ser una lista no vacía"
            )
        for i, design in enumerate(designs):
            if "name" not in design:
                raise ValueError(
                    f"Config de DesignComparisonRunner: diseño [{i}] no tiene campo 'name'"
                )
            if "reservoir" not in design:
                raise ValueError(
                    f"Config de DesignComparisonRunner: diseño '{design.get('name', i)}' "
                    f"no tiene campo 'reservoir'"
                )

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self) -> list[dict[str, Any]]:
        """
        Ejecuta el barrido multi-tarea para cada diseño, agrega los resultados
        en una tabla comparativa y la devuelve.

        Para cada diseño:
        1. Construye la config con ``build_design_config``.
        2. Instancia ``MultiTaskSweepRunner`` y llama a ``.run()``.
        3. Recoge el ``MultiTaskSweepSummary``.

        Tras ejecutar todos los diseños, llama a ``aggregate_comparison`` y
        almacena el resultado en ``self._comparison_table``.

        Returns
        -------
        list[dict[str, Any]]
            Tabla comparativa (lista de filas dict) ordenada por
            ``global_aggregate_rank`` ascendente.

        Notes
        -----
        - El ``config_id`` depende únicamente de ``spectral_radius``,
          ``input_scaling`` y ``leak_rate`` del ``config_point`` (gestionado
          por ``make_config_id`` en ``sweep_runner.py``). No depende de
          ``design_name`` ni de ``reservoir.type``, lo que permite alinear
          resultados entre diseños por ``config_id``.
        - Los resultados de cada diseño se guardan en
          ``{base_output_dir}/{design_name}/`` (gestionado por
          ``MultiTaskSweepRunner``).
        """
        design_results: list[tuple[str, MultiTaskSweepSummary]] = []

        for design in self._designs:
            design_name: str = design["name"]
            design_cfg = build_design_config(self._cfg, design)
            summary = MultiTaskSweepRunner(design_cfg).run()
            design_results.append((design_name, summary))

        self._comparison_table = self.aggregate_comparison(design_results)

        if self._comparison_table:
            csv_path, _json_path = self.save_comparison(self._comparison_table)
            print(f"comparison_summary.csv → {csv_path}")

        return self._comparison_table

    # ------------------------------------------------------------------
    # aggregate_comparison()
    # ------------------------------------------------------------------

    def aggregate_comparison(
        self,
        design_results: list[tuple[str, MultiTaskSweepSummary]],
    ) -> list[dict[str, Any]]:
        """
        Agrega los resultados de todos los diseños en una tabla comparativa.

        Parameters
        ----------
        design_results:
            Lista de ``(design_name, MultiTaskSweepSummary)`` producida por
            ``run()``.

        Returns
        -------
        list[dict[str, Any]]
            Tabla comparativa ordenada por ``global_aggregate_rank`` ascendente.
            Cada fila corresponde a una combinación ``(design_name, config_id)``.

        Columnas mínimas
        ----------------
        - ``design_name``, ``reservoir_type``, ``config_id``
        - ``spectral_radius``, ``input_scaling``, ``leak_rate``
        - ``n_seeds``
        - ``narma10_val_nmse_mean``, ``narma10_test_nmse_mean``
        - ``mg_val_nmse_mean``, ``mg_test_nmse_mean``
        - ``mc_total_mean``
        - Ranks internos: ``rank_narma10_within_design``, ``rank_mg_within_design``,
          ``rank_mc_within_design``, ``aggregate_rank_within_design``
        - Ranks globales: ``global_rank_narma10``, ``global_rank_mg``,
          ``global_rank_mc``, ``global_aggregate_rank``
        - Deltas vs baseline (para filas no-baseline con mismo ``config_id``):
          ``delta_vs_baseline_narma10_val_nmse``, ``delta_vs_baseline_mg_val_nmse``,
          ``delta_vs_baseline_mc_total``
        - Diagnósticos agregados (si disponibles): ``diag_*_mean``

        Requisitos: 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 8.4
        """
        import numpy as np
        from scipy.stats import rankdata

        # ------------------------------------------------------------------
        # 1. Construir filas base
        # ------------------------------------------------------------------
        rows: list[dict[str, Any]] = []

        # Mapa design_name → design dict (para extraer reservoir.type)
        design_map: dict[str, dict[str, Any]] = {
            d["name"]: d for d in self._designs
        }

        for design_name, summary in design_results:
            design = design_map.get(design_name, {})
            reservoir_type: str = design.get("reservoir", {}).get("type", "unknown")

            for entry in summary.configs:
                cp = entry.config_point
                row: dict[str, Any] = {
                    "design_name": design_name,
                    "reservoir_type": reservoir_type,
                    "config_id": entry.config_id,
                    "spectral_radius": cp.get("spectral_radius", None),
                    "input_scaling": cp.get("input_scaling", None),
                    "leak_rate": cp.get("leak_rate", None),
                    "n_seeds": entry.n_seeds,
                    "narma10_val_nmse_mean": entry.narma10_val_mean,
                    "narma10_test_nmse_mean": entry.narma10_test_mean,
                    "mg_val_nmse_mean": entry.mg_val_mean,
                    "mg_test_nmse_mean": entry.mg_test_mean,
                    "mc_total_mean": entry.mc_total_mean,
                    # Within-design ranks (inherited from MultiTaskSweepSummary)
                    "rank_narma10_within_design": entry.rank_narma10,
                    "rank_mg_within_design": entry.rank_mg,
                    "rank_mc_within_design": entry.rank_mc,
                    "aggregate_rank_within_design": entry.aggregate_rank,
                }
                rows.append(row)

        if not rows:
            return rows

        # ------------------------------------------------------------------
        # 2. Calcular ranks globales (method="min" para empates deterministas)
        # ------------------------------------------------------------------
        n10_vals = np.array([r["narma10_val_nmse_mean"] for r in rows], dtype=float)
        mg_vals  = np.array([r["mg_val_nmse_mean"]      for r in rows], dtype=float)
        mc_vals  = np.array([r["mc_total_mean"]          for r in rows], dtype=float)

        # narma10 y mg: menor NMSE = mejor → rank ascendente
        global_rank_n10 = rankdata(n10_vals, method="min").astype(int)
        global_rank_mg  = rankdata(mg_vals,  method="min").astype(int)
        # mc: mayor MC = mejor → rank descendente (negar valores)
        global_rank_mc  = rankdata(-mc_vals, method="min").astype(int)

        global_agg_rank = (global_rank_n10 + global_rank_mg + global_rank_mc) / 3.0

        for i, row in enumerate(rows):
            row["global_rank_narma10"]   = int(global_rank_n10[i])
            row["global_rank_mg"]        = int(global_rank_mg[i])
            row["global_rank_mc"]        = int(global_rank_mc[i])
            row["global_aggregate_rank"] = float(global_agg_rank[i])

        # ------------------------------------------------------------------
        # 3. Calcular deltas vs random_sparse_baseline
        # ------------------------------------------------------------------
        # Construir índice: config_id → fila baseline
        baseline_by_config: dict[str, dict[str, Any]] = {
            r["config_id"]: r
            for r in rows
            if r["design_name"] == "random_sparse_baseline"
        }

        for row in rows:
            if row["design_name"] == "random_sparse_baseline":
                continue
            baseline = baseline_by_config.get(row["config_id"])
            if baseline is None:
                continue
            # Negativo = mejora para NMSE; positivo = mejora para MC
            row["delta_vs_baseline_narma10_val_nmse"] = (
                row["narma10_val_nmse_mean"] - baseline["narma10_val_nmse_mean"]
            )
            row["delta_vs_baseline_mg_val_nmse"] = (
                row["mg_val_nmse_mean"] - baseline["mg_val_nmse_mean"]
            )
            row["delta_vs_baseline_mc_total"] = (
                row["mc_total_mean"] - baseline["mc_total_mean"]
            )

        # ------------------------------------------------------------------
        # 4. Agregar diagnósticos desde JSONs individuales de narma10/runs/
        # ------------------------------------------------------------------
        diag_keys = [
            "spectral_radius",
            "mean_abs_eigenvalue",
            "spectral_norm",
            "frobenius_norm",
            "density",
            "henrici_departure",
        ]
        diag_col_map = {k: f"diag_{k}_mean" for k in diag_keys}

        # Intentar leer diagnósticos para cada (design_name, config_id)
        diag_data: dict[tuple[str, str], dict[str, list[float]]] = {}

        for design_name, _summary in design_results:
            design_output_dir = self._output_dir / design_name
            narma10_runs_dir = design_output_dir / "narma10" / "runs"
            if not narma10_runs_dir.exists():
                continue
            for json_file in sorted(narma10_runs_dir.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                diag = data.get("reservoir_diagnostics")
                if not diag:
                    continue
                config_id = data.get("config_id", "")
                key = (design_name, config_id)
                if key not in diag_data:
                    diag_data[key] = {k: [] for k in diag_keys}
                for k in diag_keys:
                    val = diag.get(k)
                    if val is not None:
                        diag_data[key][k].append(float(val))

        # Añadir columnas diag_*_mean si hay datos disponibles
        has_diag = bool(diag_data)
        if has_diag:
            for row in rows:
                key = (row["design_name"], row["config_id"])
                entry_diag = diag_data.get(key)
                if entry_diag:
                    for k, col in diag_col_map.items():
                        vals = entry_diag.get(k, [])
                        row[col] = float(np.mean(vals)) if vals else None
                else:
                    for col in diag_col_map.values():
                        row[col] = None

        # ------------------------------------------------------------------
        # 5. Ordenar por global_aggregate_rank ascendente
        # ------------------------------------------------------------------
        rows.sort(key=lambda r: r["global_aggregate_rank"])

        return rows

    # ------------------------------------------------------------------
    # save_comparison()
    # ------------------------------------------------------------------

    def save_comparison(
        self,
        comparison_table: list[dict[str, Any]],
    ) -> tuple[Path, Path]:
        """
        Persiste la tabla comparativa en ``{output_dir}/comparison_summary.csv``
        y ``{output_dir}/comparison_summary.json``.

        Parameters
        ----------
        comparison_table:
            Lista de filas dict producida por ``aggregate_comparison``.
            Se asume que ya está ordenada por ``global_aggregate_rank``.

        Returns
        -------
        tuple[Path, Path]
            ``(csv_path, json_path)``

        CSV
        ---
        - Todas las filas de ``comparison_table`` como CSV plano.
        - ``fieldnames`` derivados de las claves de la primera fila (captura
          columnas opcionales como ``diag_*`` y ``delta_*``).
        - ``extrasaction="ignore"`` para robustez ante columnas variables.

        JSON
        ----
        - ``"table"``: lista completa de filas.
        - ``"best_overall"``: fila con menor ``global_aggregate_rank``
          (primera fila tras el ordenado).
        - ``"best_by_design"``: dict ``design_name`` → mejor fila por
          ``aggregate_rank_within_design`` (mínimo) para ese diseño.
        - ``"timestamp"``: ISO 8601 UTC.
        - ``"n_designs"``: número de diseños distintos.
        - ``"n_configs_per_design"``: total de filas / n_designs.

        Requisitos: 5.9, 5.10, 5.11
        """
        if not comparison_table:
            raise ValueError("save_comparison: comparison_table está vacía")

        # Crear directorio de salida si no existe
        self._output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self._output_dir / "comparison_summary.csv"
        json_path = self._output_dir / "comparison_summary.json"

        # ------------------------------------------------------------------
        # CSV
        # ------------------------------------------------------------------
        preferred_order = [
            "design_name",
            "reservoir_type",
            "config_id",
            "spectral_radius",
            "input_scaling",
            "leak_rate",
            "n_seeds",
            "narma10_val_nmse_mean",
            "narma10_test_nmse_mean",
            "mg_val_nmse_mean",
            "mg_test_nmse_mean",
            "mc_total_mean",
            "rank_narma10_within_design",
            "rank_mg_within_design",
            "rank_mc_within_design",
            "aggregate_rank_within_design",
            "global_rank_narma10",
            "global_rank_mg",
            "global_rank_mc",
            "global_aggregate_rank",
            "delta_vs_baseline_narma10_val_nmse",
            "delta_vs_baseline_mg_val_nmse",
            "delta_vs_baseline_mc_total",
            "diag_spectral_radius_mean",
            "diag_mean_abs_eigenvalue_mean",
            "diag_spectral_norm_mean",
            "diag_frobenius_norm_mean",
            "diag_density_mean",
            "diag_henrici_departure_mean",
        ]

        all_keys = set()
        for row in comparison_table:
            all_keys.update(row.keys())

        fieldnames = [k for k in preferred_order if k in all_keys]
        fieldnames += sorted(k for k in all_keys if k not in fieldnames)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(comparison_table)

        # ------------------------------------------------------------------
        # JSON
        # ------------------------------------------------------------------
        # best_overall: primera fila (ya ordenada por global_aggregate_rank asc)
        best_overall = comparison_table[0]

        # best_by_design: para cada design_name, la fila con menor
        # aggregate_rank_within_design
        best_by_design: dict[str, dict[str, Any]] = {}
        for row in comparison_table:
            design_name = row["design_name"]
            current_best = best_by_design.get(design_name)
            if current_best is None or (
                row["aggregate_rank_within_design"]
                < current_best["aggregate_rank_within_design"]
            ):
                best_by_design[design_name] = row

        # Metadatos
        n_designs = len(best_by_design)
        n_configs_per_design = (
            len(comparison_table) / n_designs if n_designs > 0 else 0
        )
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        json_payload: dict[str, Any] = {
            "timestamp": timestamp,
            "n_designs": n_designs,
            "n_configs_per_design": n_configs_per_design,
            "best_overall": best_overall,
            "best_by_design": best_by_design,
            "table": comparison_table,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False, default=str)

        return csv_path, json_path
