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


def reservoir_diagnostics(W: np.ndarray) -> dict[str, float]:
    """
    Agrega todas las métricas de diagnóstico de W en un diccionario.

    Parámetros
    ----------
    W : np.ndarray, shape (N, N)
        Matriz recurrente cuadrada.

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
    """
    return {
        "spectral_radius": compute_spectral_radius(W),
        "mean_abs_eigenvalue": compute_mean_abs_eigenvalue(W),
        "spectral_norm": compute_spectral_norm(W),
        "frobenius_norm": compute_frobenius_norm(W),
        "density": compute_density(W),
        "henrici_departure": compute_henrici_departure(W),
    }
