import numpy as np


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Mean Squared Error.

        NMSE = MSE(y_true, y_pred) / Var(y_true)

    Devuelve 0.0 para predicción perfecta.
    """
    var = np.var(y_true)
    if var == 0:
        raise ValueError("Var(y_true) == 0: NMSE no está definido")
    return float(np.mean((y_true - y_pred) ** 2) / var)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
