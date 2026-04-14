import numpy as np

from rc_lab.models.esn import ESNModel
from rc_lab.readouts.ridge import RidgeReadout
from rc_lab.utils.seeding import set_seed


class MemoryCapacityEvaluator:
    """
    Evalúa la Memory Capacity total de una ESN ya construida.

        MC = sum_{k=1}^{max_delay} r²(k)

    donde r²(k) es el coeficiente de determinación entre u(t-k) e y_k(t),
    con y_k(t) obtenido entrenando un readout ridge para cada delay k.

    Se invoca de forma independiente al ExperimentRunner estándar.
    """

    def __init__(
        self,
        max_delay: int = 50,
        ridge_param: float = 1e-6,
        washout: int = 200,
    ) -> None:
        self.max_delay = max_delay
        self.ridge_param = ridge_param
        self.washout = washout

    def evaluate(
        self,
        esn: ESNModel,
        u: np.ndarray,   # señal de entrada U(-1, 1), shape (T, 1)
        seed: int,
    ) -> float:
        """
        Calcula la MC total.

        Parámetros
        ----------
        esn  : ESNModel ya construido (W, Win, bias fijados)
        u    : señal de entrada, shape (T, 1)
        seed : semilla para reproducibilidad

        Devuelve
        --------
        MC total: escalar finito, no negativo, normalmente acotado por esn.N
        """
        set_seed(seed)
        T = u.shape[0]

        # Obtener estados del reservoir
        X_all, _ = esn.run_states(u, washout=self.washout)
        u_post = u[self.washout:].ravel()  # (T - washout,)
        T_eff = X_all.shape[0]

        mc_total = 0.0
        for k in range(1, self.max_delay + 1):
            if k >= T_eff:
                break

            # Target: u(t - k) para t en [washout, T)
            y_k = u_post[: T_eff - k]          # (T_eff - k,)
            X_k = X_all[k : T_eff]             # (T_eff - k, N)

            readout = RidgeReadout(ridge_param=self.ridge_param)
            readout.fit(X_k, y_k.reshape(-1, 1))
            y_pred = readout.predict(X_k).ravel()

            # r²(k) = coeficiente de determinación
            ss_res = np.sum((y_k - y_pred) ** 2)
            ss_tot = np.sum((y_k - np.mean(y_k)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mc_total += max(0.0, r2)  # clamp a 0 por estabilidad numérica

        return float(mc_total)
