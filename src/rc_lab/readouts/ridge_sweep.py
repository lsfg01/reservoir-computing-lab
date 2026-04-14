from dataclasses import dataclass

import numpy as np

from rc_lab.metrics.error import nmse
from rc_lab.readouts.ridge import RidgeReadout


@dataclass
class RidgeSelectionResult:
    best_ridge: float
    best_val_nmse: float
    val_curve: dict[float, float]  # ridge_param → val_nmse


class RidgeParamSelector:
    """
    Selecciona ridge_param evaluando candidatos sobre el split de validación.

    Reutiliza las design matrices F_train y F_val ya construidas — no
    recomputa el reservoir ni los estados del ESN.
    """

    def __init__(self, candidates: list[float]) -> None:
        if not candidates:
            raise ValueError("La lista de candidatos no puede estar vacía")
        self._candidates = candidates

    def select(
        self,
        F_train: np.ndarray,
        Y_train: np.ndarray,
        F_val: np.ndarray,
        Y_val: np.ndarray,
    ) -> RidgeSelectionResult:
        """
        Evalúa cada candidato de ridge_param sobre (F_val, Y_val).

        Para cada alpha: entrena RidgeReadout(alpha) sobre (F_train, Y_train)
        y calcula NMSE sobre (F_val, Y_val). Devuelve el alpha con menor NMSE.

        La interfaz no acepta F_test ni Y_test — el test nunca participa
        en la selección.
        """
        val_curve: dict[float, float] = {}
        for alpha in self._candidates:
            readout = RidgeReadout(ridge_param=alpha)
            readout.fit(F_train, Y_train)
            y_pred_val = readout.predict(F_val)
            val_curve[alpha] = nmse(Y_val, y_pred_val)

        best_ridge = min(val_curve, key=val_curve.__getitem__)
        return RidgeSelectionResult(
            best_ridge=best_ridge,
            best_val_nmse=val_curve[best_ridge],
            val_curve=val_curve,
        )
