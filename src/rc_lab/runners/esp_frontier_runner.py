"""
ESPFrontierRunner — Estudio de frontera ESP (Paso 0 de la campaña).

Barre el producto (family × s_in × alpha × rho × seed) midiendo el
washout empírico vía evaluate_esp_sampled y diagnósticos de W. Produce
una tabla washout(rho) lista para graficar.

Guardado incremental: un JSON por punto, summary JSON al finalizar.
"""

from __future__ import annotations

import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rc_lab.metrics.stability import evaluate_esp_sampled
from rc_lab.models.esn import ESNModel
from rc_lab.reservoirs.diagnostics import compute_spectral_norm, compute_spectral_radius
from rc_lab.runners.runner import resolve_reservoir
from rc_lab.utils.io import make_json_safe
from rc_lab.utils.timing import timer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _final_slope_negative(d_curve: np.ndarray, frac: float = 0.05) -> bool:
    """
    Devuelve True si d_curve seguía decreciendo al final de T.

    Compara la media del último `frac` de puntos con la media del tramo
    previo de igual tamaño. Permite distinguir truncamiento (pendiente
    negativa, podría sincronizar si T fuera mayor) de divergencia.
    """
    T = len(d_curve)
    window = max(1, int(T * frac))
    last = float(np.mean(d_curve[-window:]))
    prev = float(np.mean(d_curve[-2 * window: -window])) if T >= 2 * window else float(np.mean(d_curve[:window]))
    return last < prev


def _csv_value(value: Any) -> Any:
    safe = make_json_safe(value)
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, ensure_ascii=False, allow_nan=False)
    return "" if safe is None else safe


def _subsample_curve(curve: np.ndarray, step: int = 10) -> list[float]:
    """Devuelve la curva submuestreada cada `step` pasos, siempre incluyendo el último."""
    indices = list(range(0, len(curve), step))
    if (len(curve) - 1) not in indices:
        indices.append(len(curve) - 1)
    return [float(curve[i]) for i in indices]


# ---------------------------------------------------------------------------
# Punto de datos individual
# ---------------------------------------------------------------------------

def _point_key(family_name: str, s_in: float, alpha: float, rho: float) -> str:
    return f"{family_name}_sin{s_in:.4f}_a{alpha:.4f}_rho{rho:.4f}"


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

class ESPFrontierRunner:
    """
    Orquesta el estudio de frontera ESP.

    Parámetros del config (dict cargado desde YAML):
        frontier.name         : nombre del experimento
        frontier.output_dir   : directorio de salida
        frontier.seeds        : lista de semillas enteras
        esp.T                 : longitud de secuencia para evaluate_esp_sampled
        esp.n_pairs           : número de pares de condiciones iniciales
        esp.eps               : umbral de sincronización
        families              : lista de dicts con {name, type, N, ...kwargs}
        grid.spectral_radius  : lista de rho
        grid.input_scaling    : lista de s_in
        grid.leak_rate        : lista de alpha (normalmente [1.0])
        diagnostics.transient_kmax : (ignorado; reservado para extensiones futuras)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._validate(config)

        frontier = config["frontier"]
        self._name: str = frontier["name"]
        self._output_dir = Path(frontier["output_dir"])
        self._seeds: list[int] = list(frontier["seeds"])

        esp = config["esp"]
        self._T: int = int(esp["T"])
        self._n_pairs: int = int(esp["n_pairs"])
        self._eps: float = float(esp["eps"])

        grid = config["grid"]
        self._rho_list: list[float] = list(grid["spectral_radius"])
        self._sin_list: list[float] = list(grid["input_scaling"])
        self._alpha_list: list[float] = list(grid["leak_rate"])

        self._families: list[dict[str, Any]] = list(config["families"])

        diagnostics = config.get("diagnostics", {})
        self._sat_window_frac: float = float(diagnostics.get("saturation_window_frac", 0.5))

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def _validate(self, cfg: dict[str, Any]) -> None:
        for key in ("frontier", "esp", "families", "grid"):
            if key not in cfg:
                raise ValueError(f"Falta la clave requerida en config: {key!r}")

        frontier = cfg["frontier"]
        for k in ("name", "output_dir", "seeds"):
            if k not in frontier:
                raise ValueError(f"Falta frontier.{k} en config")
        if not isinstance(frontier["seeds"], list) or len(frontier["seeds"]) == 0:
            raise ValueError("frontier.seeds debe ser una lista no vacía")

        esp = cfg["esp"]
        for k in ("T", "n_pairs", "eps"):
            if k not in esp:
                raise ValueError(f"Falta esp.{k} en config")
        if int(esp["T"]) < 1:
            raise ValueError(f"esp.T debe ser >= 1, recibido {esp['T']}")
        if int(esp["n_pairs"]) < 1:
            raise ValueError(f"esp.n_pairs debe ser >= 1, recibido {esp['n_pairs']}")
        if not (0.0 < float(esp["eps"]) < 1.0):
            raise ValueError(f"esp.eps debe estar en (0, 1), recibido {esp['eps']}")

        grid = cfg["grid"]
        for k in ("spectral_radius", "input_scaling", "leak_rate"):
            if k not in grid:
                raise ValueError(f"Falta grid.{k} en config")
            if not isinstance(grid[k], list) or len(grid[k]) == 0:
                raise ValueError(f"grid.{k} debe ser una lista no vacía")

        families = cfg["families"]
        if not isinstance(families, list) or len(families) == 0:
            raise ValueError("families debe ser una lista no vacía")
        for fam in families:
            for k in ("name", "type", "N"):
                if k not in fam:
                    raise ValueError(f"Cada familia debe tener {k!r}; falta en {fam}")

    # ------------------------------------------------------------------
    # Expansión del grid
    # ------------------------------------------------------------------

    def expand_points(self) -> list[dict[str, Any]]:
        """
        Devuelve la lista de puntos a evaluar:
        (family, s_in, alpha, rho) × seeds.
        """
        points = []
        for fam in self._families:
            for s_in, alpha, rho in itertools.product(
                self._sin_list, self._alpha_list, self._rho_list
            ):
                for seed in self._seeds:
                    points.append({
                        "family": fam,
                        "s_in": s_in,
                        "alpha": alpha,
                        "rho": rho,
                        "seed": seed,
                    })
        return points

    def n_points(self) -> int:
        return len(self._families) * len(self._sin_list) * len(self._alpha_list) * len(self._rho_list) * len(self._seeds)

    # ------------------------------------------------------------------
    # Evaluación de un punto individual
    # ------------------------------------------------------------------

    def _eval_point(self, point: dict[str, Any]) -> dict[str, Any]:
        fam = point["family"]
        s_in: float = point["s_in"]
        alpha: float = point["alpha"]
        rho: float = point["rho"]
        seed: int = point["seed"]
        N: int = int(fam["N"])

        # Construir res_cfg: excluir 'name' y 'N'; añadir spectral_radius e input_scaling
        extra_kwargs = {
            k: v for k, v in fam.items()
            if k not in ("name", "N")
        }
        res_cfg: dict[str, Any] = {
            **extra_kwargs,
            "spectral_radius": rho,
            "input_scaling": s_in,
        }
        # leak_rate NO se pasa a resolve_reservoir
        builder = resolve_reservoir(res_cfg)
        matrices = builder.build(N=N, n_inputs=1, seed=seed)

        esn = ESNModel(matrices.W, matrices.Win, matrices.bias, leak_rate=alpha)

        # Saturation diagnosis: same u as evaluate_esp_sampled (first draw from seed)
        rng_sat = np.random.default_rng(seed)
        u_sat = rng_sat.uniform(-1.0, 1.0, (self._T, 1))
        states_sat, _ = esn.run_states(u_sat, washout=0, x0=np.zeros(N))
        window_size = max(1, int(self._T * self._sat_window_frac))
        states_window = states_sat[-window_size:]
        saturation_mean = float(np.mean(np.abs(states_window)))
        saturation_frac = float(np.mean(np.abs(states_window) > 0.99))

        with timer() as t_esp:
            esp = evaluate_esp_sampled(esn, seed=seed, T=self._T, n_pairs=self._n_pairs, eps=self._eps)
        esp_elapsed = t_esp["elapsed"]

        sigma_max = compute_spectral_norm(matrices.W)
        rho_real = compute_spectral_radius(matrices.W)

        # Flag de pendiente final para pares no sincronizados
        unsync_final_slope_neg = [
            _final_slope_negative(pr.d_curve)
            for pr in esp.pair_results
            if not pr.synchronized
        ]
        n_unsync = len(unsync_final_slope_neg)
        n_unsync_descending = sum(unsync_final_slope_neg)
        frac_unsync_decreasing = (
            float(n_unsync_descending / n_unsync) if n_unsync > 0 else None
        )

        # d_curve_mean submuestreada
        d_curve_sub = _subsample_curve(esp.d_curve_mean, step=10)

        return {
            "family_name": fam["name"],
            "family_type": fam["type"],
            "N": N,
            "s_in": s_in,
            "alpha": alpha,
            "rho_target": rho,
            "seed": seed,
            "fraction_synchronized": esp.fraction_synchronized,
            "sync_time_mean": esp.sync_time_mean,
            "sync_time_std": esp.sync_time_std,
            "frac_unsync_decreasing": frac_unsync_decreasing,
            "n_unsync": n_unsync,
            "n_unsync_descending": n_unsync_descending,
            "saturation_mean": saturation_mean,
            "saturation_frac": saturation_frac,
            "sigma_max": sigma_max,
            "rho_real": rho_real,
            "d_curve_mean_sub": d_curve_sub,
            "esp_elapsed_s": esp_elapsed,
            "T": self._T,
            "n_pairs": self._n_pairs,
            "eps": self._eps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Persistencia incremental
    # ------------------------------------------------------------------

    def _save_point(self, result: dict[str, Any]) -> Path:
        runs_dir = self._output_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        key = _point_key(result["family_name"], result["s_in"], result["alpha"], result["rho_target"])
        fname = f"{key}_seed{result['seed']}.json"
        path = runs_dir / fname
        safe = make_json_safe(result)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2, allow_nan=False)
        return path

    # ------------------------------------------------------------------
    # Agregación
    # ------------------------------------------------------------------

    def _aggregate(self, all_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Agrega sobre seeds: por (family_name, s_in, alpha, rho_target) devuelve
        media/std de sync_time_mean y fraction_synchronized, y media de sigma_max/rho_real.
        """
        from collections import defaultdict
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in all_results:
            key = (r["family_name"], r["s_in"], r["alpha"], r["rho_target"])
            groups[key].append(r)

        rows = []
        for (family_name, s_in, alpha, rho_target), recs in sorted(groups.items()):
            frac_sync_vals = [r["fraction_synchronized"] for r in recs]
            sigma_max_vals = [r["sigma_max"] for r in recs]
            rho_real_vals = [r["rho_real"] for r in recs]

            sync_time_vals = [r["sync_time_mean"] for r in recs if r["sync_time_mean"] is not None]

            # Saturation: average over seeds
            saturation_mean_mean = float(np.mean([r["saturation_mean"] for r in recs]))
            saturation_frac_mean = float(np.mean([r["saturation_frac"] for r in recs]))

            # Truncation flag: pool non-sync pairs across all seeds
            total_unsync = sum(r["n_unsync"] for r in recs)
            total_descending = sum(r["n_unsync_descending"] for r in recs)
            nonsync_fraction_descending = (
                float(total_descending / total_unsync) if total_unsync > 0 else None
            )

            row: dict[str, Any] = {
                "family_name": family_name,
                "s_in": s_in,
                "alpha": alpha,
                "rho_target": rho_target,
                "n_seeds": len(recs),
                "fraction_synchronized_mean": float(np.mean(frac_sync_vals)),
                "fraction_synchronized_std": float(np.std(frac_sync_vals)),
                "sync_time_mean_mean": float(np.mean(sync_time_vals)) if sync_time_vals else None,
                "sync_time_mean_std": float(np.std(sync_time_vals)) if len(sync_time_vals) > 1 else None,
                "sigma_max_mean": float(np.mean(sigma_max_vals)),
                "rho_real_mean": float(np.mean(rho_real_vals)),
                "saturation_mean_mean": saturation_mean_mean,
                "saturation_frac_mean": saturation_frac_mean,
                "nonsync_fraction_descending": nonsync_fraction_descending,
            }
            rows.append(row)

        return {
            "name": self._name,
            "T": self._T,
            "n_pairs": self._n_pairs,
            "eps": self._eps,
            "seeds": self._seeds,
            "rows": rows,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _save_summary(self, summary: dict[str, Any]) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self._output_dir / "summary.json"
        safe = make_json_safe(summary)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2, allow_nan=False)

        rows = summary.get("rows", [])
        if rows:
            csv_path = self._output_dir / "summary.csv"
            fieldnames: list[str] = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: _csv_value(v) for k, v in row.items()})

        return json_path

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Ejecuta el barrido completo. Guarda cada punto incrementalmente.
        Devuelve el summary agregado.
        """
        points = self.expand_points()
        total = len(points)
        all_results: list[dict[str, Any]] = []

        print(f"[ESP Frontier] {self._name}: {total} puntos ({len(self._families)} familias × "
              f"{len(self._sin_list)} s_in × {len(self._alpha_list)} alpha × "
              f"{len(self._rho_list)} rho × {len(self._seeds)} seeds)")

        for i, point in enumerate(points, 1):
            result = self._eval_point(point)
            self._save_point(result)
            frac = result["fraction_synchronized"]
            st = result["sync_time_mean"]
            st_str = f"{st:.1f}" if st is not None else "N/A"
            print(
                f"  [{i:>4}/{total}] {result['family_name']} "
                f"rho={result['rho_target']:.3f} s_in={result['s_in']:.3f} "
                f"alpha={result['alpha']:.2f} seed={result['seed']:>4} "
                f"| frac_sync={frac:.2f} sync_time={st_str}"
            )
            all_results.append(result)

        summary = self._aggregate(all_results)
        self._save_summary(summary)
        return summary
