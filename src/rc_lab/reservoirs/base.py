from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ReservoirMatrices:
    W: np.ndarray    # (N, N)        matriz recurrente
    Win: np.ndarray  # (N, n_inputs) matriz de entrada
    bias: np.ndarray # (N,)          bias del reservoir


class BaseReservoirBuilder(ABC):
    """Contrato mínimo para cualquier constructor de reservoir."""

    @abstractmethod
    def build(self, N: int, n_inputs: int, seed: int) -> ReservoirMatrices:
        """
        Construye las matrices del reservoir.

        Parámetros
        ----------
        N        : tamaño del reservoir (número de nodos)
        n_inputs : dimensión de la entrada
        seed     : semilla para reproducibilidad

        Devuelve ReservoirMatrices con W (N,N), Win (N, n_inputs), bias (N,).
        """
        ...
