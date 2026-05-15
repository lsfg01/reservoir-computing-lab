from dataclasses import dataclass

import numpy as np

from rc_lab.models.esn import ESNModel
from rc_lab.readouts.ridge import RidgeReadout
from rc_lab.tasks.delay_recall import build_delay_recall_targets, delay_recall_valid_start


@dataclass
class MCResult:
    """Resultado detallado de la evaluación de Memory Capacity."""
    mc_total: float
    mc_by_delay: np.ndarray  # shape (kmax_eff,), mc_by_delay[k-1] = MC_k
    kmax: int                # kmax efectivo usado
    fit_samples: int
    eval_samples: int


class MemoryCapacityEvaluator:
    """
    Evalúa la Linear Memory Capacity (MC) de una ESN según el benchmark estándar.

        MC = sum_{k=1}^{kmax} MC_k

    donde MC_k = corr²(u(t-k), ŷ_k(t)) calculado out-of-sample en el bloque eval.

    Protocolo
    ---------
    1. Genera u ~ U(-1, 1) internamente con un RNG local (sin efectos globales).
    2. Aplica un único reset + washout inicial del reservoir.
    3. Construye el target multisalida Y[:, k-1] = u(t-k) para k=1..kmax.
    4. Divide secuencialmente en bloques fit / eval.
    5. Entrena un único readout ridge multisalida sobre el bloque fit.
    6. Evalúa corr² por delay en el bloque eval (out-of-sample).

    Parámetros
    ----------
    washout      : pasos iniciales descartados para eliminar transitorio.
    input_length : longitud total de u generada (incluyendo washout).
    fit_fraction : fracción de T_valid usada para entrenar el readout.
    kmax         : número máximo de delays evaluados.
                   Si None, se usa la heurística 2 * esn.N, acotada por datos.

                   Nota metodológica: cuando se comparan configuraciones con el
                   mismo N, el default 2*N produce un kmax comparable. Si en el
                   futuro se comparan reservoirs con distinto N (e.g. N=100 vs
                   N=500), conviene fijar kmax explícitamente en config para
                   garantizar que mc_total se calcula sobre el mismo número de
                   delays en todas las configuraciones y los valores son
                   estrictamente comparables.

    ridge_param  : regularización del readout ridge.
    """

    def __init__(
        self,
        washout: int = 200,
        input_length: int = 3000,
        fit_fraction: float = 0.5,
        kmax: int | None = None,
        ridge_param: float = 1e-6,
    ) -> None:
        if not 0.0 < fit_fraction < 1.0:
            raise ValueError("fit_fraction debe estar en (0, 1)")
        self.washout = washout
        self.input_length = input_length
        self.fit_fraction = fit_fraction
        self.kmax = kmax
        self.ridge_param = ridge_param

    def evaluate(self, esn: ESNModel, seed: int) -> float:
        """Devuelve mc_total (escalar). Interfaz pública principal."""
        return self.evaluate_details(esn, seed).mc_total

    def evaluate_details(self, esn: ESNModel, seed: int) -> MCResult:
        """Devuelve MCResult completo, incluyendo mc_by_delay."""
        # Generador local — sin efectos sobre el estado global de numpy
        rng = np.random.default_rng(seed)
        u = rng.uniform(-1.0, 1.0, (self.input_length, 1))

        # Estados del reservoir tras washout
        X_all, _ = esn.run_states(u, washout=self.washout, x0=None)
        u_post = u[self.washout:].ravel()   # (T_eff,)
        T_eff = X_all.shape[0]

        # kmax efectivo: heurística o valor explícito, acotado por datos
        if self.kmax is not None:
            kmax_eff = min(self.kmax, T_eff - 1)
        else:
            kmax_eff = min(2 * esn.N, T_eff - 1)

        if kmax_eff < 1:
            raise ValueError(
                f"kmax_eff={kmax_eff}: no hay suficientes datos tras washout "
                f"(T_eff={T_eff}, washout={self.washout})"
            )

        # Número de muestras válidas (alineadas con el delay máximo)
        T_valid = T_eff - kmax_eff
        if T_valid < 2:
            raise ValueError(
                f"T_valid={T_valid} insuficiente para split fit/eval "
                f"(T_eff={T_eff}, kmax_eff={kmax_eff})"
            )

        # Construir design matrix X y target multisalida Y
        # X[t] = estado del reservoir en el paso t + kmax_eff (post-washout)
        # Y[t, k-1] = u(t + kmax_eff - k) = u retardado k pasos respecto a X[t]
        X = X_all[kmax_eff : kmax_eff + T_valid]          # (T_valid, N)
        valid_start = delay_recall_valid_start(self.washout, kmax_eff)
        Y_full = build_delay_recall_targets(u, kmax_eff)
        Y = Y_full[valid_start : valid_start + T_valid]

        # Split secuencial fit / eval
        n_fit = max(1, int(T_valid * self.fit_fraction))
        n_eval = T_valid - n_fit
        if n_eval < 1:
            raise ValueError(
                f"Bloque eval vacío: T_valid={T_valid}, fit_fraction={self.fit_fraction}"
            )

        X_fit,  Y_fit  = X[:n_fit],  Y[:n_fit]
        X_eval, Y_eval = X[n_fit:],  Y[n_fit:]

        # Único readout ridge multisalida entrenado sobre fit
        readout = RidgeReadout(ridge_param=self.ridge_param)
        readout.fit(X_fit, Y_fit)

        # Predicción out-of-sample en eval
        Y_pred = readout.predict(X_eval)   # (n_eval, kmax_eff)

        # MC_k = corr²(target_k, pred_k) en bloque eval
        mc_by_delay = np.empty(kmax_eff)
        for k in range(kmax_eff):
            target = Y_eval[:, k]
            pred   = Y_pred[:, k]
            var_t = np.var(target)
            var_p = np.var(pred)
            if var_t < 1e-12 or var_p < 1e-12:
                mc_by_delay[k] = 0.0
            else:
                cov = np.mean((target - target.mean()) * (pred - pred.mean()))
                mc_by_delay[k] = max(0.0, cov ** 2 / (var_t * var_p))

        mc_total = float(np.sum(mc_by_delay))

        return MCResult(
            mc_total=mc_total,
            mc_by_delay=mc_by_delay,
            kmax=kmax_eff,
            fit_samples=n_fit,
            eval_samples=n_eval,
        )
