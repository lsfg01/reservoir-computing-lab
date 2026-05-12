"""
CycleJumpReservoir — reservoir cíclico con conexiones de salto.

Topología: ciclo dirigido base + conexiones de salto regulares.
  - Ciclo base:  W[(i+1)%N, i] = cycle_weight  para i in range(N)
  - Saltos:      W[(i+j)%N, i] += jump_weight  para cada j en jumps, i in range(N)

Convención de W: W[row, col] = transmisión de la neurona col a la neurona row.
"""

from __future__ import annotations

import warnings

import numpy as np

from rc_lab.reservoirs.base import BaseReservoirBuilder, ReservoirMatrices


class CycleJumpReservoir(BaseReservoirBuilder):
    """
    Reservoir cíclico con conexiones de salto y reescalado espectral.

    Parámetros del constructor
    --------------------------
    spectral_radius : radio espectral objetivo de W tras el reescalado (>= 0).
    input_scaling   : escala de Win; entradas en U(-input_scaling, input_scaling).
    cycle_weight    : peso de las conexiones del ciclo base antes del reescalado.
    jumps           : entero o lista de enteros positivos que definen los
                      desplazamientos de las conexiones de salto.
    jump_weight     : peso de las conexiones de salto antes del reescalado.
    bias_scaling    : escala del bias; 0.0 desactiva el bias.

    Notas
    -----
    - ``jumps`` se normaliza internamente a ``list[int]``.
    - Cada j en jumps debe ser un entero positivo (j > 0); se lanza ValueError
      en caso contrario.
    - Las advertencias sobre auto-bucles, coincidencia con el ciclo base y
      saltos congruentes módulo N se emiten en ``build()``, ya que N sólo se
      conoce en ese momento.
    """

    def __init__(
        self,
        spectral_radius: float,
        input_scaling: float,
        cycle_weight: float = 1.0,
        jumps: int | list[int] = 3,
        jump_weight: float = 0.3,
        bias_scaling: float = 0.0,
    ) -> None:
        # --- Validaciones del constructor ---
        if spectral_radius < 0:
            raise ValueError(
                f"spectral_radius debe ser >= 0, se recibió {spectral_radius}."
            )
        if input_scaling < 0:
            raise ValueError(
                f"input_scaling debe ser >= 0, se recibió {input_scaling}."
            )
        if bias_scaling < 0:
            raise ValueError(
                f"bias_scaling debe ser >= 0, se recibió {bias_scaling}."
            )

        # Normalizar jumps a list[int]
        if isinstance(jumps, int):
            jumps_list: list[int] = [jumps]
        else:
            jumps_list = list(jumps)

        # Validar que cada j es un entero positivo
        for j in jumps_list:
            if not isinstance(j, (int, np.integer)) or j <= 0:
                raise ValueError(
                    f"Cada elemento de jumps debe ser un entero positivo (j > 0), "
                    f"se recibió j={j!r}."
                )

        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.cycle_weight = cycle_weight
        self.jumps: list[int] = [int(j) for j in jumps_list]
        self.jump_weight = jump_weight
        self.bias_scaling = bias_scaling

    def build(self, N: int, n_inputs: int, seed: int) -> ReservoirMatrices:
        """
        Construye las matrices del reservoir.

        Parámetros
        ----------
        N        : tamaño del reservoir (número de nodos). Debe ser > 0.
        n_inputs : dimensión de la entrada. Debe ser > 0.
        seed     : semilla para reproducibilidad.

        Devuelve
        --------
        ReservoirMatrices con W (N, N), Win (N, n_inputs), bias (N,).

        Lanza
        -----
        ValueError
            Si N <= 0, n_inputs <= 0, o si spectral_radius > 0 y el radio
            espectral de W antes del reescalado es cero.
        """
        if N <= 0:
            raise ValueError(f"N debe ser > 0, se recibió {N}.")
        if n_inputs <= 0:
            raise ValueError(f"n_inputs debe ser > 0, se recibió {n_inputs}.")

        # --- Advertencias dependientes de N (emitidas aquí porque N es necesario) ---
        self._emit_jump_warnings(N)

        rng = np.random.default_rng(seed)

        # --- Construir W ---
        W = np.zeros((N, N), dtype=float)

        # Ciclo base: neurona i transmite a (i+1)%N
        for i in range(N):
            W[(i + 1) % N, i] = self.cycle_weight

        # Saltos: para cada j, neurona i transmite a (i+j)%N
        for j in self.jumps:
            for i in range(N):
                W[(i + j) % N, i] += self.jump_weight

        # --- Reescalado espectral ---
        if self.spectral_radius == 0.0:
            W = np.zeros((N, N), dtype=float)
        else:
            eigenvalues = np.linalg.eigvals(W)
            rho_actual = float(np.max(np.abs(eigenvalues)))
            if rho_actual == 0.0:
                raise ValueError(
                    "El radio espectral de W antes del reescalado es cero, "
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

    def _emit_jump_warnings(self, N: int) -> None:
        """Emite advertencias sobre propiedades de los saltos dado N."""
        # Advertencia 1: auto-bucles (j % N == 0)
        for j in self.jumps:
            if j % N == 0:
                warnings.warn(
                    f"El salto j={j} satisface j % N == 0 (N={N}): genera "
                    f"auto-bucles en W[i, i]. Esto puede alterar la dinámica "
                    f"del reservoir de forma no deseada.",
                    UserWarning,
                    stacklevel=3,
                )

        # Advertencia 2: coincidencia con el ciclo base (j % N == 1)
        for j in self.jumps:
            if j % N == 1:
                warnings.warn(
                    f"El salto j={j} satisface j % N == 1 (N={N}): coincide "
                    f"con el ciclo base y no añade una dirección estructural "
                    f"nueva a W.",
                    UserWarning,
                    stacklevel=3,
                )

        # Advertencia 3: varios jumps congruentes módulo N
        residues = [j % N for j in self.jumps]
        seen: set[int] = set()
        duplicates: set[int] = set()
        for r in residues:
            if r in seen:
                duplicates.add(r)
            seen.add(r)
        if duplicates:
            warnings.warn(
                f"Varios elementos de jumps={self.jumps} son congruentes "
                f"módulo N={N} (residuos duplicados: {sorted(duplicates)}). "
                f"Pueden producirse conexiones solapadas en W.",
                UserWarning,
                stacklevel=3,
            )
