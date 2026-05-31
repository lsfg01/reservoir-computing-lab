"""
Evaluador empírico de la Echo State Property (ESP) via sincronización de estados.

La ESP exige que el estado actual quede determinado por el historial de entrada,
no por la condición inicial. Empíricamente: con la misma entrada u(t) desde dos
estados iniciales distintos x0, x0', las trayectorias deben sincronizar
(‖x(t) − x'(t)‖₂ → 0). El tiempo hasta sincronización es el washout empírico.

Nota de condicionalidad: el washout empírico es condicional al input u. Distintas
realizaciones de u (distinta seed) pueden producir tiempos de sincronización
distintos; los agregados de esta función reflejan la distribución sobre n_pairs
pares de condiciones iniciales bajo una única realización de u.

Uso previsto:
- Diagnóstico por config: registrar sync_time_mean y fraction_synchronized junto
  a ρ, Henrici y crecimiento transitorio.
- Barrido de ρ: llamar variando spectral_radius para construir washout(ρ).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rc_lab.models.esn import ESNModel


@dataclass
class PairSyncResult:
    """Resultado de sincronización para un par de condiciones iniciales."""
    d_curve: np.ndarray   # distancia relativa d(t) = ‖x(t)−x'(t)‖/‖x(0)−x'(0)‖, shape (T,)
    sync_time: int | None  # primer t con d(t) < eps; None si no sincroniza
    synchronized: bool


@dataclass
class ESPResult:
    """Resultado agregado del evaluador de ESP empírica."""
    pair_results: list[PairSyncResult]
    fraction_synchronized: float
    sync_time_mean: float | None   # media sobre pares sincronizados; None si ninguno
    sync_time_std: float | None    # desviación; None si ninguno o solo uno
    d_curve_mean: np.ndarray       # curva d(t) promediada sobre todos los pares, shape (T,)
    n_pairs: int
    T: int
    eps: float


def sync_pair(
    esn: ESNModel,
    u: np.ndarray,
    x0: np.ndarray,
    x0p: np.ndarray,
    eps: float = 1e-3,
) -> PairSyncResult:
    """
    Evalúa la sincronización de estados para un par de condiciones iniciales.

    Evoluciona el reservoir desde x0 y desde x0' bajo la misma entrada u, y
    mide la distancia relativa d(t) = ‖x(t)−x'(t)‖₂ / ‖x(0)−x'(0)‖₂.

    Parámetros
    ----------
    esn  : modelo ESN ya construido (W, Win, bias, leak_rate)
    u    : entrada compartida, shape (T, n_inputs)
    x0   : condición inicial primera, shape (N,)
    x0p  : condición inicial segunda, shape (N,)
    eps  : umbral relativo para declarar sincronización

    Devuelve
    --------
    PairSyncResult con d_curve (T pasos), sync_time (o None) y synchronized.
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    T = u.shape[0]
    d0 = float(np.linalg.norm(x0 - x0p))
    if d0 == 0.0:
        # Pares idénticos: trivialmente sincronizados desde t=0
        d_curve = np.zeros(T)
        return PairSyncResult(d_curve=d_curve, sync_time=0, synchronized=True)

    # Evolucionar ambas trayectorias con washout=0 para obtener todos los estados
    states_a, _ = esn.run_states(u, washout=0, x0=x0)   # (T, N)
    states_b, _ = esn.run_states(u, washout=0, x0=x0p)  # (T, N)

    diff_norms = np.linalg.norm(states_a - states_b, axis=1)  # (T,)
    d_curve = diff_norms / d0

    sync_time: int | None = None
    for t in range(T):
        if d_curve[t] < eps:
            sync_time = t
            break

    return PairSyncResult(
        d_curve=d_curve,
        sync_time=sync_time,
        synchronized=(sync_time is not None),
    )


def evaluate_esp(
    esn: ESNModel,
    u: np.ndarray,
    x0_pairs: list[tuple[np.ndarray, np.ndarray]],
    eps: float = 1e-3,
) -> ESPResult:
    """
    Evalúa la ESP empírica sobre una lista de pares de condiciones iniciales.

    Parámetros
    ----------
    esn      : modelo ESN ya construido
    u        : entrada compartida, shape (T, n_inputs) o (T,)
    x0_pairs : lista de tuplas (x0, x0') de condiciones iniciales
    eps      : umbral relativo para declarar sincronización

    Devuelve
    --------
    ESPResult con resultados por par y estadísticas agregadas.
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if len(x0_pairs) == 0:
        raise ValueError("x0_pairs no puede estar vacío")

    T = u.shape[0]
    pair_results = [sync_pair(esn, u, x0, x0p, eps=eps) for x0, x0p in x0_pairs]

    sync_times = [r.sync_time for r in pair_results if r.synchronized]
    fraction_synchronized = len(sync_times) / len(pair_results)

    sync_time_mean: float | None = None
    sync_time_std: float | None = None
    if sync_times:
        sync_time_mean = float(np.mean(sync_times))
        sync_time_std = float(np.std(sync_times)) if len(sync_times) > 1 else None

    # Curva promedio sobre todos los pares (incluidos los no sincronizados)
    d_curves = np.stack([r.d_curve for r in pair_results], axis=0)  # (n_pairs, T)
    d_curve_mean = np.mean(d_curves, axis=0)

    return ESPResult(
        pair_results=pair_results,
        fraction_synchronized=fraction_synchronized,
        sync_time_mean=sync_time_mean,
        sync_time_std=sync_time_std,
        d_curve_mean=d_curve_mean,
        n_pairs=len(pair_results),
        T=T,
        eps=eps,
    )


def evaluate_esp_sampled(
    esn: ESNModel,
    seed: int,
    T: int = 2000,
    n_pairs: int = 5,
    eps: float = 1e-3,
) -> ESPResult:
    """
    Evaluador de alto nivel: muestrea pares de condiciones iniciales y una
    entrada aleatoria, luego llama a evaluate_esp y agrega resultados.

    Las condiciones iniciales se muestrean como x0, x0' ~ U(-1, 1)^N de forma
    independiente (sin perturbación local). La entrada u ~ U(-1, 1)^T sigue el
    mismo proceso que delay_recall/MC para comparabilidad.

    El washout empírico es condicional al input u generado con esta seed.

    Parámetros
    ----------
    esn     : modelo ESN ya construido
    seed    : semilla para RNG local (reproduce resultados deterministas)
    T       : longitud de la secuencia de entrada
    n_pairs : número de pares de condiciones iniciales
    eps     : umbral relativo para declarar sincronización

    Devuelve
    --------
    ESPResult completo con estadísticas agregadas.
    """
    if T < 1:
        raise ValueError(f"T debe ser >= 1, recibido T={T}")
    if n_pairs < 1:
        raise ValueError(f"n_pairs debe ser >= 1, recibido n_pairs={n_pairs}")
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps debe estar en (0, 1), recibido eps={eps}")

    rng = np.random.default_rng(seed)
    N = esn.N

    u = rng.uniform(-1.0, 1.0, (T, 1))

    x0_pairs = [
        (rng.uniform(-1.0, 1.0, N), rng.uniform(-1.0, 1.0, N))
        for _ in range(n_pairs)
    ]

    return evaluate_esp(esn, u, x0_pairs, eps=eps)
