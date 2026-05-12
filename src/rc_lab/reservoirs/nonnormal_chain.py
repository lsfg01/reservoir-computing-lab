import numpy as np

from rc_lab.reservoirs.base import BaseReservoirBuilder, ReservoirMatrices


class NonnormalChainReservoir(BaseReservoirBuilder):
    """
    Reservoir no-normal con topología de cadena.

    La matriz recurrente W se construye como:

        W = spectral_radius * I + chain_strength * S

    donde S es el shift unidireccional: S[i+1, i] = 1 para i = 0, ..., N-2.

    Propiedades matemáticas:
    - W es triangular inferior.
    - Todos los autovalores son iguales a spectral_radius, por lo que
      spectral_radius(W) = |spectral_radius| por construcción.
    - chain_strength modifica la no-normalidad sin cambiar los autovalores.
    - Para chain_strength > 0, la matriz es no-normal: ||W^k||_2 puede crecer
      transitoriamente antes de decaer.
    - henrici_departure(W) > 0 cuando chain_strength > 0.

    Parámetros del constructor
    --------------------------
    spectral_radius : controla los autovalores de W (todos iguales a d).
                      Debe ser >= 0.
    input_scaling   : escala de Win; entradas en U(-input_scaling, input_scaling).
                      Debe ser >= 0.
    chain_strength  : peso del shift S; controla la no-normalidad.
                      Debe ser >= 0. Default: 0.3.
    bias_scaling    : escala del bias; 0.0 desactiva el bias.
                      Debe ser >= 0. Default: 0.0.

    Notas
    -----
    - NO se reescala W por radio espectral tras la construcción.
    - spectral_radius(W) = |d| = |spectral_radius| por construcción.
    - chain_strength modifica la no-normalidad sin cambiar los autovalores.
    - Convención de W: W[row, col] = transmisión de neurona col a neurona row.
      Por tanto W[i+1, i] = chain_strength transmite de la neurona i a la i+1.
    """

    def __init__(
        self,
        spectral_radius: float,
        input_scaling: float,
        chain_strength: float = 0.3,
        bias_scaling: float = 0.0,
    ) -> None:
        if spectral_radius < 0:
            raise ValueError(
                f"spectral_radius debe ser >= 0, se recibió {spectral_radius}"
            )
        if input_scaling < 0:
            raise ValueError(
                f"input_scaling debe ser >= 0, se recibió {input_scaling}"
            )
        if chain_strength < 0:
            raise ValueError(
                f"chain_strength debe ser >= 0, se recibió {chain_strength}"
            )
        if bias_scaling < 0:
            raise ValueError(
                f"bias_scaling debe ser >= 0, se recibió {bias_scaling}"
            )

        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.chain_strength = chain_strength
        self.bias_scaling = bias_scaling

    def build(self, N: int, n_inputs: int, seed: int) -> ReservoirMatrices:
        """
        Construye las matrices del reservoir de cadena no-normal.

        Parámetros
        ----------
        N        : tamaño del reservoir (número de nodos), debe ser > 0.
        n_inputs : dimensión de la entrada, debe ser > 0.
        seed     : semilla para reproducibilidad.

        Devuelve ReservoirMatrices con W (N,N), Win (N, n_inputs), bias (N,).

        Construcción de W:
        1. W = spectral_radius * I  (diagonal)
        2. W[i+1, i] = chain_strength  para i in range(N-1)  (shift unidireccional)
        NO se reescala W; spectral_radius(W) = spectral_radius por construcción.
        """
        if N <= 0:
            raise ValueError(f"N debe ser > 0, se recibió {N}")
        if n_inputs <= 0:
            raise ValueError(f"n_inputs debe ser > 0, se recibió {n_inputs}")

        rng = np.random.default_rng(seed)

        # --- W: d*I + g*S ---
        d = self.spectral_radius
        W = d * np.eye(N, dtype=float)

        # Shift unidireccional: W[i+1, i] = chain_strength
        for i in range(N - 1):
            W[i + 1, i] = self.chain_strength

        # --- Win: uniforme en [-input_scaling, input_scaling] ---
        Win = rng.uniform(-self.input_scaling, self.input_scaling, (N, n_inputs))

        # --- bias ---
        if self.bias_scaling > 0:
            bias = rng.uniform(-self.bias_scaling, self.bias_scaling, (N,))
        else:
            bias = np.zeros(N)

        return ReservoirMatrices(W=W, Win=Win, bias=bias)
