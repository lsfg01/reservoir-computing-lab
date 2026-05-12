"""
Diagnósticos numéricos de la matriz recurrente W.

Todas las funciones son puras (sin efectos secundarios) y operan sobre
arrays de NumPy. Se pueden aplicar a cualquier familia de reservoir.
"""

import numpy as np


def compute_spectral_radius(W: np.ndarray) -> float:
    """
    Radio espectral de W: max(|autovalores(W)|).

    Parámetros
    ----------
    W : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.

    Devuelve
    --------
    float
        Máximo módulo de los autovalores de W.
    """
    eigvals = np.linalg.eigvals(W)
    return float(np.max(np.abs(eigvals)))


def compute_mean_abs_eigenvalue(W: np.ndarray) -> float:
    """
    Media de los módulos de los autovalores de W: mean(|autovalores(W)|).

    Parámetros
    ----------
    W : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.

    Devuelve
    --------
    float
        Media de los módulos de los autovalores de W.
    """
    eigvals = np.linalg.eigvals(W)
    return float(np.mean(np.abs(eigvals)))


def compute_spectral_norm(W: np.ndarray) -> float:
    """
    Norma espectral de W: mayor valor singular (sigma_max).

    Equivalente a np.linalg.norm(W, ord=2).

    Parámetros
    ----------
    W : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.

    Devuelve
    --------
    float
        Mayor valor singular de W.
    """
    return float(np.linalg.norm(W, ord=2))


def compute_frobenius_norm(W: np.ndarray) -> float:
    """
    Norma de Frobenius de W: sqrt(sum W_ij^2).

    Parámetros
    ----------
    W : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.

    Devuelve
    --------
    float
        Norma de Frobenius de W.
    """
    return float(np.linalg.norm(W, "fro"))


def compute_density(W: np.ndarray, tol: float = 1e-12) -> float:
    """
    Densidad de W: fracción de entradas con |W[i,j]| > tol.

    Para un CycleReservoir de tamaño N, devuelve exactamente 1/N
    (N entradas no nulas en una matriz N×N).

    Parámetros
    ----------
    W   : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.
    tol : float, opcional
        Umbral de tolerancia para considerar una entrada como no nula.
        Por defecto 1e-12.

    Devuelve
    --------
    float
        Fracción de entradas con |W[i,j]| > tol sobre el total N*N.
    """
    total = W.size
    nonzero = int(np.sum(np.abs(W) > tol))
    return nonzero / total


def compute_henrici_departure(W: np.ndarray) -> float:
    """
    Medida de no-normalidad de Henrici.

    dep_F(W) = sqrt(max(0, ||W||_F^2 - sum|lambda_i|^2)) / ||W||_F

    Devuelve 0.0 si ||W||_F == 0 (matriz nula).

    Para matrices normales (e.g. ciclo puro, matrices unitarias), el valor
    es exactamente 0 (o muy próximo a 0 por errores de punto flotante).

    Parámetros
    ----------
    W : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.

    Devuelve
    --------
    float
        Medida de no-normalidad de Henrici en [0, 1).
        0.0 indica que W es normal.
    """
    frob = float(np.linalg.norm(W, "fro"))
    if frob == 0.0:
        return 0.0

    eigvals = np.linalg.eigvals(W)
    sum_sq_abs_eig = float(np.sum(np.abs(eigvals) ** 2))
    frob_sq = frob ** 2

    departure = np.sqrt(max(0.0, frob_sq - sum_sq_abs_eig)) / frob
    return float(departure)


def singular_values(W: np.ndarray) -> np.ndarray:
    """
    Valores singulares de W en orden descendente.

    Usa ``np.linalg.svd(W, compute_uv=False)``, que ya devuelve los valores
    singulares ordenados de mayor a menor.

    Parámetros
    ----------
    W : np.ndarray, shape (M, N)
        Matriz (no necesariamente cuadrada).

    Devuelve
    --------
    np.ndarray
        Array 1-D con los valores singulares en orden descendente.
    """
    return np.linalg.svd(W, compute_uv=False)


def singular_value_max(W: np.ndarray) -> float:
    """
    Mayor valor singular de W (norma espectral).

    Parámetros
    ----------
    W : np.ndarray
        Matriz de entrada.

    Devuelve
    --------
    float
        Valor singular máximo.
    """
    return float(np.max(singular_values(W)))


def singular_value_min(W: np.ndarray) -> float:
    """
    Menor valor singular de W.

    Parámetros
    ----------
    W : np.ndarray
        Matriz de entrada.

    Devuelve
    --------
    float
        Valor singular mínimo.
    """
    return float(np.min(singular_values(W)))


def singular_value_mean(W: np.ndarray) -> float:
    """
    Media aritmética de los valores singulares de W.

    Parámetros
    ----------
    W : np.ndarray
        Matriz de entrada.

    Devuelve
    --------
    float
        Media de los valores singulares.
    """
    return float(np.mean(singular_values(W)))


def singular_value_q90(W: np.ndarray) -> float:
    """
    Percentil 90 de los valores singulares de W.

    Parámetros
    ----------
    W : np.ndarray
        Matriz de entrada.

    Devuelve
    --------
    float
        Percentil 90 de los valores singulares.
    """
    return float(np.percentile(singular_values(W), 90))


def singular_condition_number(W: np.ndarray, eps: float = 1e-12) -> float:
    """
    Número de condición singular: sigma_max / sigma_min.

    Devuelve ``float('inf')`` si ``sigma_min < eps`` (matriz singular o
    casi singular).

    Parámetros
    ----------
    W   : np.ndarray
        Matriz de entrada.
    eps : float, opcional
        Umbral por debajo del cual sigma_min se considera cero.
        Por defecto 1e-12.

    Devuelve
    --------
    float
        Ratio sigma_max / sigma_min, o ``inf`` si sigma_min < eps.
    """
    sv = singular_values(W)
    sigma_max = float(np.max(sv))
    sigma_min = float(np.min(sv))
    if sigma_min < eps:
        return float("inf")
    return sigma_max / sigma_min


def transient_growth_curve(W: np.ndarray, kmax: int) -> list[float]:
    """
    Calcula ||W^k||_2 para k=1,...,kmax mediante multiplicación iterativa.

    Parámetros
    ----------
    W    : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.
    kmax : int
        Número de pasos. Debe ser > 0.

    Devuelve
    --------
    list[float]
        Lista de longitud kmax con ||W^k||_2 para k=1,...,kmax.

    Lanza
    -----
    ValueError
        Si W no es cuadrada o si kmax <= 0.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W debe ser una matriz cuadrada")
    if kmax <= 0:
        raise ValueError(f"kmax debe ser > 0, se recibió {kmax}")

    curve = []
    Wk = W.copy()
    for k in range(1, kmax + 1):
        norm_k = float(np.linalg.norm(Wk, ord=2))
        curve.append(norm_k)
        if k < kmax:
            Wk = Wk @ W
    return curve


def transient_growth_max(W: np.ndarray, kmax: int) -> float:
    """
    Máximo de ||W^k||_2 para k=1,...,kmax.

    Parámetros
    ----------
    W    : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.
    kmax : int
        Número de pasos. Debe ser > 0.

    Devuelve
    --------
    float
        Máximo valor de transient_growth_curve(W, kmax).
    """
    curve = transient_growth_curve(W, kmax)
    return float(np.max(curve))


def transient_growth_argmax(W: np.ndarray, kmax: int) -> int:
    """
    Primer k donde se alcanza el máximo de ||W^k||_2.

    Parámetros
    ----------
    W    : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.
    kmax : int
        Número de pasos. Debe ser > 0.

    Devuelve
    --------
    int
        Entero en [1, kmax] donde se alcanza el máximo de transient_growth_curve(W, kmax).
    """
    curve = transient_growth_curve(W, kmax)
    return int(np.argmax(curve)) + 1  # +1 porque curve[0] corresponde a k=1


def reservoir_diagnostics(W: np.ndarray, transient_kmax: int = 50) -> dict[str, float]:
    """
    Agrega todas las métricas de diagnóstico de W en un diccionario.

    Parámetros
    ----------
    W             : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.
    transient_kmax : int, opcional
        Número de pasos para calcular ``transient_growth_curve``.
        Por defecto 50.

    Devuelve
    --------
    dict[str, float]
        Diccionario con exactamente las claves:
        - ``spectral_radius``
        - ``mean_abs_eigenvalue``
        - ``spectral_norm``
        - ``frobenius_norm``
        - ``density``
        - ``henrici_departure``
        - ``singular_value_max``
        - ``singular_value_min``
        - ``singular_value_mean``
        - ``singular_value_q90``
        - ``singular_condition_number``
        - ``transient_growth_max``
        - ``transient_growth_argmax``
    """
    sv = singular_values(W)
    curve = transient_growth_curve(W, transient_kmax)

    sigma_max = float(np.max(sv))
    sigma_min = float(np.min(sv))
    if sigma_min < 1e-12:
        cond = float(np.inf)
    else:
        cond = sigma_max / sigma_min

    tgm = float(np.max(curve))
    tgm_argmax = int(np.argmax(curve)) + 1

    return {
        "spectral_radius": compute_spectral_radius(W),
        "mean_abs_eigenvalue": compute_mean_abs_eigenvalue(W),
        "spectral_norm": compute_spectral_norm(W),
        "frobenius_norm": compute_frobenius_norm(W),
        "density": compute_density(W),
        "henrici_departure": compute_henrici_departure(W),
        "singular_value_max": sigma_max,
        "singular_value_min": sigma_min,
        "singular_value_mean": float(np.mean(sv)),
        "singular_value_q90": float(np.percentile(sv, 90)),
        "singular_condition_number": cond,
        "transient_growth_max": tgm,
        "transient_growth_argmax": tgm_argmax,
    }
