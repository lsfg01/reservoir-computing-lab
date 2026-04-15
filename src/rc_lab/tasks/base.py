from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

StatePolicy = Literal["reset", "carryover"]


@dataclass
class TaskData:
    # --- Splits con warmup incluido ---
    # u_train / y_train incluyen los primeros `washout` pasos de calentamiento.
    # El runner debe descartar esos pasos con run_states(..., washout=washout)
    # y luego alinear con y_train[washout:] para obtener Y_train puntuado.
    u_train: np.ndarray   # (washout + n_train, n_inputs)
    y_train: np.ndarray   # (washout + n_train, n_outputs)

    # --- Splits puntuados (post-warmup) ---
    # u_val / y_val / u_test / y_test son ya los arrays puntuados.
    # El runner los usa directamente como Y_val, Y_test sin slicing adicional.
    u_val:  np.ndarray | None  # (n_val, n_inputs)  — None si n_val == 0
    y_val:  np.ndarray | None  # (n_val, n_outputs) — None si n_val == 0
    u_test: np.ndarray         # (n_test, n_inputs)
    y_test: np.ndarray         # (n_test, n_outputs)

    washout: int

    # --- Política de estado entre splits (metadato de trazabilidad) ---
    # Generado por la task según su configuración. El runner bifurca
    # su lógica en función de u_val_full / u_test_full, no de este campo.
    state_policy: StatePolicy = "carryover"

    # --- Bloques completos con warmup para val/test (solo si reset) ---
    # Presentes cuando state_policy == "reset".
    # shape: (washout + n_val, n_inputs) y (washout + n_test, n_inputs).
    # None cuando state_policy == "carryover" o n_val == 0.
    u_val_full:  np.ndarray | None = None
    u_test_full: np.ndarray | None = None


class BaseTask(ABC):
    """Contrato mínimo que toda tarea estándar debe cumplir."""

    @abstractmethod
    def generate(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        """
        Genera los datos de la tarea y devuelve un TaskData.

        Cuando n_val == 0, TaskData.u_val y TaskData.y_val serán None,
        preservando la compatibilidad con el ExperimentRunner existente.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador de la tarea (e.g. 'narma10')."""
        ...

    @property
    @abstractmethod
    def primary_metric(self) -> str:
        """Métrica principal de evaluación (e.g. 'nmse')."""
        ...
