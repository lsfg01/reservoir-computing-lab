import numpy as np

from rc_lab.reservoirs.base import BaseReservoirBuilder, ReservoirMatrices


class CycleReservoir(BaseReservoirBuilder):
    """
    Reservoir con topología de ciclo dirigido puro.

    La matriz recurrente W tiene exactamente N entradas no nulas, ubicadas en
    las posiciones W[(i+1) % N, i] = cycle_weight para todo i en range(N).
    Tras construir el ciclo, W se reescala para que su radio espectral sea
    exactamente spectral_radius.

    Parámetros del constructor
    --------------------------
    spectral_radius : radio espectral objetivo de W (>= 0).
    input_scaling   : escala de Win; entradas en U(-input_scaling, input_scaling).
    cycle_weight    : peso de las conexiones del ciclo antes del reescalado.
    bias_scaling    : escala del bias; 0.0 desactiva el bias.

    Nota: leak_rate NO es parámetro del constructor. Vive en el config_point
    del grid y se pasa a ESNModel directamente.
    """

    def __init__(
        self,
        spectral_radius: float,
        input_scaling: float,
        cycle_weight: float = 1.0,
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
        if bias_scaling < 0:
            raise ValueError(
                f"bias_scaling debe ser >= 0, se recibió {bias_scaling}"
            )

        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.cycle_weight = cycle_weight
        self.bias_scaling = bias_scaling

    def build(self, N: int, n_inputs: int, seed: int) -> ReservoirMatrices:
        """
        Construye las matrices del reservoir cíclico.

        Parámetros
        ----------
        N        : tamaño del reservoir (número de nodos), debe ser > 0.
        n_inputs : dimensión de la entrada, debe ser > 0.
        seed     : semilla para reproducibilidad.

        Devuelve ReservoirMatrices con W (N,N), Win (N, n_inputs), bias (N,).
        """
        if N <= 0:
            raise ValueError(f"N debe ser > 0, se recibió {N}")
        if n_inputs <= 0:
            raise ValueError(f"n_inputs debe ser > 0, se recibió {n_inputs}")

        rng = np.random.default_rng(seed)

        # --- W: ciclo dirigido ---
        # W[(i+1) % N, i] = cycle_weight  para i in range(N)
        # Convención: W[row, col] = transmisión de neurona col a neurona row.
        W = np.zeros((N, N), dtype=float)
        for i in range(N):
            W[(i + 1) % N, i] = self.cycle_weight

        # --- Reescalado espectral ---
        if self.spectral_radius == 0.0:
            # Devolver W nula sin intentar el reescalado
            W = np.zeros((N, N), dtype=float)
        else:
            rho_actual = float(np.max(np.abs(np.linalg.eigvals(W))))
            if rho_actual == 0.0:
                raise ValueError(
                    "El radio espectral de W antes del reescalado es 0, "
                    "pero spectral_radius > 0. No es posible reescalar una "
                    "matriz sin dinámica recurrente a un radio espectral positivo."
                )
            W = W * (self.spectral_radius / rho_actual)

        # --- Win ---
        Win = rng.uniform(-self.input_scaling, self.input_scaling, (N, n_inputs))

        # --- bias ---
        if self.bias_scaling > 0:
            bias = rng.uniform(-self.bias_scaling, self.bias_scaling, (N,))
        else:
            bias = np.zeros(N)

        return ReservoirMatrices(W=W, Win=Win, bias=bias)
