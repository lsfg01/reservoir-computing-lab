import numpy as np


class ESNModel:
    """
    Dinámica del reservoir ESN con leak rate.

    Ecuación de actualización:
        x(t) = (1 - alpha) * x(t-1) + alpha * tanh(W @ x(t-1) + Win @ u(t) + bias)

    Para alpha = 1.0 se reduce al caso estándar sin leak:
        x(t) = tanh(W @ x(t-1) + Win @ u(t) + bias)
    """

    def __init__(
        self,
        W: np.ndarray,
        Win: np.ndarray,
        bias: np.ndarray,
        leak_rate: float = 1.0,
    ) -> None:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W debe ser una matriz cuadrada 2D")
        if Win.ndim != 2 or Win.shape[0] != W.shape[0]:
            raise ValueError("Win.shape[0] debe coincidir con W.shape[0]")
        if bias.ndim != 1 or bias.shape[0] != W.shape[0]:
            raise ValueError("bias.shape[0] debe coincidir con W.shape[0]")
        if not (0 < leak_rate <= 1.0):
            raise ValueError("leak_rate debe estar en (0, 1]")

        self._W = W
        self._Win = Win
        self._bias = bias
        self._alpha = leak_rate

    def run_states(
        self,
        u: np.ndarray,
        washout: int = 0,
        x0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta la dinámica del reservoir sobre la secuencia de entrada u.

        Parámetros
        ----------
        u       : array (T, n_inputs)
        washout : número de pasos iniciales a descartar
        x0      : estado inicial (N,); si None se usan ceros

        Devuelve
        --------
        X_out   : estados post-washout, shape (T - washout, N)
        x_final : último estado x(T-1), shape (N,)
        """
        T = u.shape[0]
        N = self._W.shape[0]
        alpha = self._alpha

        x = np.zeros(N) if x0 is None else x0.copy()
        states = np.empty((T, N))

        for t in range(T):
            x = (1.0 - alpha) * x + alpha * np.tanh(
                self._W @ x + self._Win @ u[t] + self._bias
            )
            states[t] = x

        return states[washout:], x

    @property
    def N(self) -> int:
        """Tamaño del reservoir."""
        return self._W.shape[0]
