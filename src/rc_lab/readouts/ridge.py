from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge

ReadoutMode = Literal["states", "extended"]


def build_readout_features(
    X: np.ndarray,
    u: np.ndarray,
    mode: ReadoutMode = "states",
) -> np.ndarray:
    """
    Construye la design matrix para el readout.

    Modos
    -----
    "states"   : devuelve X tal cual, shape (T, N)
    "extended" : devuelve [1, u(t), x(t)] por fila, shape (T, 1 + n_inputs + N)
                 siguiendo la formulación extendida habitual en ESN

    Parámetros
    ----------
    X    : estados del reservoir, shape (T, N)
    u    : entrada correspondiente, shape (T, n_inputs)
    mode : "states" | "extended"
    """
    if mode == "states":
        return X
    elif mode == "extended":
        T = X.shape[0]
        ones = np.ones((T, 1))
        return np.hstack([ones, u, X])
    else:
        raise ValueError(f"mode debe ser 'states' o 'extended', recibido: {mode!r}")


class RidgeReadout:
    """
    Readout lineal entrenado con ridge regression sobre la design matrix F.

        Wout = argmin ||F @ Wout.T - Y||^2 + beta * ||Wout||^2

    F es la design matrix producida por build_readout_features().
    """

    def __init__(self, ridge_param: float = 1e-6) -> None:
        if ridge_param <= 0:
            raise ValueError("ridge_param debe ser > 0")
        self._ridge_param = ridge_param
        self._model: Ridge | None = None

    def fit(self, F: np.ndarray, Y: np.ndarray) -> None:
        """Entrena el readout sobre la design matrix F."""
        if F.shape[0] != Y.shape[0]:
            raise ValueError("F y Y deben tener el mismo número de filas")
        self._model = Ridge(alpha=self._ridge_param, fit_intercept=False)
        self._model.fit(F, Y)

    def predict(self, F: np.ndarray) -> np.ndarray:
        """Aplica el readout a una design matrix F. Devuelve shape (T, n_outputs)."""
        if self._model is None:
            raise RuntimeError("readout not fitted")
        out = self._model.predict(F)
        return out.reshape(F.shape[0], -1)

    @property
    def Wout(self) -> np.ndarray:
        """Matriz de pesos del readout, shape (n_outputs, d_features)."""
        if self._model is None:
            raise RuntimeError("readout not fitted")
        coef = self._model.coef_
        return coef.reshape(1, -1) if coef.ndim == 1 else coef
