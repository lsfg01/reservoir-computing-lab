from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TaskData:
    u_train: np.ndarray        # (T_train, n_inputs)
    y_train: np.ndarray        # (T_train, n_outputs)
    u_val: np.ndarray | None   # (T_val, n_inputs)  — None si n_val == 0
    y_val: np.ndarray | None   # (T_val, n_outputs) — None si n_val == 0
    u_test: np.ndarray         # (T_test,  n_inputs)
    y_test: np.ndarray         # (T_test,  n_outputs)
    washout: int


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
