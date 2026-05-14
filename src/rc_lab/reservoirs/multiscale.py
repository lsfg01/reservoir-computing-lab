from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import random as sparse_random

from rc_lab.reservoirs.base import BaseReservoirBuilder, ReservoirMatrices

_SUPPORTED_BLOCK_TYPES = {"random_sparse"}
_SUPPORTED_PRESETS = {"two_scale_random", "three_scale_random"}


class MultiScaleReservoir(BaseReservoirBuilder):
    """
    Reservoir multiescala por bloques.

    Cada bloque diagonal se construye como un random sparse con radio espectral
    local ``spectral_radius * rho_factor``. Los bloques vecinos pueden conectarse
    mediante acoplamiento débil sin reescalado global posterior.
    """

    def __init__(
        self,
        spectral_radius: float,
        input_scaling: float,
        preset: str | None = None,
        block_specs: list[dict[str, Any]] | None = None,
        sparsity: float = 0.9,
        coupling_strength: float = 0.02,
        coupling_density: float = 0.1,
        coupling_mode: str = "adjacent",
        coupling_direction: str = "bidirectional",
        bias_scaling: float = 0.0,
    ) -> None:
        if preset is None and block_specs is None:
            raise ValueError("MultiScaleReservoir requiere 'preset' o 'block_specs'.")
        if preset is not None and block_specs is not None:
            raise ValueError("'preset' y 'block_specs' son mutuamente excluyentes.")
        if spectral_radius < 0:
            raise ValueError(f"spectral_radius debe ser >= 0, se recibió {spectral_radius}.")
        if input_scaling < 0:
            raise ValueError(f"input_scaling debe ser >= 0, se recibió {input_scaling}.")
        if sparsity < 0 or sparsity > 1:
            raise ValueError(f"sparsity debe estar en [0, 1], se recibió {sparsity}.")
        if bias_scaling < 0:
            raise ValueError(f"bias_scaling debe ser >= 0, se recibió {bias_scaling}.")
        if coupling_strength < 0:
            raise ValueError(
                f"coupling_strength debe ser >= 0, se recibió {coupling_strength}."
            )
        if not 0 <= coupling_density <= 1:
            raise ValueError(
                f"coupling_density debe estar en [0, 1], se recibió {coupling_density}."
            )
        if coupling_mode != "adjacent":
            raise ValueError(
                "coupling_mode soportado en esta versión: 'adjacent'. "
                f"Se recibió {coupling_mode!r}."
            )
        if coupling_direction not in {"forward", "backward", "bidirectional"}:
            raise ValueError(
                "coupling_direction debe ser 'forward', 'backward' o 'bidirectional'. "
                f"Se recibió {coupling_direction!r}."
            )

        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.preset = preset
        self.sparsity = sparsity
        self.coupling_strength = coupling_strength
        self.coupling_density = coupling_density
        self.coupling_mode = coupling_mode
        self.coupling_direction = coupling_direction
        self.bias_scaling = bias_scaling
        self.block_specs = self._normalize_block_specs(preset, block_specs, sparsity)

    def build(self, N: int, n_inputs: int, seed: int) -> ReservoirMatrices:
        if N <= 0:
            raise ValueError(f"N debe ser > 0, se recibió {N}.")
        if n_inputs <= 0:
            raise ValueError(f"n_inputs debe ser > 0, se recibió {n_inputs}.")

        rng = np.random.default_rng(seed)
        sizes = self._block_sizes(N, self.block_specs)
        W = np.zeros((N, N), dtype=float)

        starts: list[int] = []
        offset = 0
        for spec, size in zip(self.block_specs, sizes):
            starts.append(offset)
            rho_block = self.spectral_radius * spec["rho_factor"]
            W_block = self._build_random_sparse_block(
                size=size,
                spectral_radius=rho_block,
                sparsity=spec["sparsity"],
                rng=rng,
            )
            W[offset:offset + size, offset:offset + size] = W_block
            offset += size

        self._add_adjacent_coupling(W, starts, sizes, rng)

        Win = rng.uniform(-self.input_scaling, self.input_scaling, (N, n_inputs))
        if self.bias_scaling > 0:
            bias = rng.uniform(-self.bias_scaling, self.bias_scaling, (N,))
        else:
            bias = np.zeros(N)

        return ReservoirMatrices(W=W, Win=Win, bias=bias)

    @staticmethod
    def _block_sizes(N: int, block_specs: list[dict[str, Any]]) -> list[int]:
        sizes: list[int] = []
        for spec in block_specs[:-1]:
            sizes.append(int(round(N * spec["fraction"])))
        sizes.append(N - sum(sizes))

        if any(size < 1 for size in sizes):
            raise ValueError(
                f"Todos los bloques deben tener tamaño >= 1; tamaños calculados: {sizes}."
            )
        return sizes

    @classmethod
    def _normalize_block_specs(
        cls,
        preset: str | None,
        block_specs: list[dict[str, Any]] | None,
        global_sparsity: float,
    ) -> list[dict[str, Any]]:
        if preset is not None:
            specs = cls._preset_to_block_specs(preset, global_sparsity)
        else:
            specs = [dict(spec) for spec in block_specs or []]

        cls._validate_block_specs(specs, global_sparsity)
        return specs

    @staticmethod
    def _preset_to_block_specs(preset: str, sparsity: float) -> list[dict[str, Any]]:
        if preset == "two_scale_random":
            return [
                {
                    "name": "fast",
                    "fraction": 0.5,
                    "block_type": "random_sparse",
                    "rho_factor": 0.55,
                    "sparsity": sparsity,
                },
                {
                    "name": "slow",
                    "fraction": 0.5,
                    "block_type": "random_sparse",
                    "rho_factor": 1.0,
                    "sparsity": sparsity,
                },
            ]
        if preset == "three_scale_random":
            return [
                {
                    "name": "fast",
                    "fraction": 0.3,
                    "block_type": "random_sparse",
                    "rho_factor": 0.55,
                    "sparsity": sparsity,
                },
                {
                    "name": "medium",
                    "fraction": 0.3,
                    "block_type": "random_sparse",
                    "rho_factor": 0.80,
                    "sparsity": sparsity,
                },
                {
                    "name": "slow",
                    "fraction": 0.4,
                    "block_type": "random_sparse",
                    "rho_factor": 1.0,
                    "sparsity": sparsity,
                },
            ]
        raise ValueError(
            f"Preset multiescala desconocido: {preset!r}. "
            f"Disponibles: {sorted(_SUPPORTED_PRESETS)}."
        )

    @staticmethod
    def _validate_block_specs(
        block_specs: list[dict[str, Any]],
        global_sparsity: float,
    ) -> None:
        if not block_specs:
            raise ValueError("block_specs debe ser una lista no vacía.")

        required = {"name", "fraction", "block_type", "rho_factor"}
        fractions: list[float] = []
        for i, spec in enumerate(block_specs):
            missing = required - set(spec)
            if missing:
                raise ValueError(f"block_specs[{i}] no contiene claves requeridas: {sorted(missing)}.")

            block_type = spec["block_type"]
            if block_type not in _SUPPORTED_BLOCK_TYPES:
                raise ValueError(
                    f"block_type={block_type!r} todavía no está implementado en "
                    "MultiScaleReservoir. La arquitectura está preparada para "
                    "futuros block_type, pero esta versión solo soporta "
                    "'random_sparse'."
                )

            fraction = float(spec["fraction"])
            if fraction <= 0:
                raise ValueError(f"block_specs[{i}].fraction debe ser > 0.")
            fractions.append(fraction)
            spec["fraction"] = fraction

            rho_factor = float(spec["rho_factor"])
            if rho_factor < 0:
                raise ValueError(f"block_specs[{i}].rho_factor debe ser >= 0.")
            spec["rho_factor"] = rho_factor

            sparsity = float(spec.get("sparsity", global_sparsity))
            if sparsity < 0 or sparsity > 1:
                raise ValueError(f"block_specs[{i}].sparsity debe estar en [0, 1].")
            spec["sparsity"] = sparsity

        if abs(sum(fractions) - 1.0) >= 1e-6:
            raise ValueError(
                f"Las fracciones de block_specs deben sumar 1.0; suma={sum(fractions)}."
            )

    @staticmethod
    def _build_random_sparse_block(
        size: int,
        spectral_radius: float,
        sparsity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        density = 1.0 - sparsity
        random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        W = sparse_random(
            size,
            size,
            density=density,
            format="csr",
            random_state=random_state,
            data_rvs=lambda s: rng.uniform(-1.0, 1.0, s),
        ).toarray()

        rho_actual = MultiScaleReservoir._compute_spectral_radius(W)
        if rho_actual > 0:
            W = W * (spectral_radius / rho_actual)
        return W

    @staticmethod
    def _compute_spectral_radius(W: np.ndarray) -> float:
        vals = np.linalg.eigvals(W)
        return float(np.max(np.abs(vals)))

    def _add_adjacent_coupling(
        self,
        W: np.ndarray,
        starts: list[int],
        sizes: list[int],
        rng: np.random.Generator,
    ) -> None:
        if self.coupling_strength == 0 or self.coupling_density == 0:
            return

        for i in range(len(sizes) - 1):
            src_start, src_size = starts[i], sizes[i]
            dst_start, dst_size = starts[i + 1], sizes[i + 1]

            if self.coupling_direction in {"forward", "bidirectional"}:
                self._fill_coupling_block(
                    W,
                    row_slice=slice(dst_start, dst_start + dst_size),
                    col_slice=slice(src_start, src_start + src_size),
                    rng=rng,
                )
            if self.coupling_direction in {"backward", "bidirectional"}:
                self._fill_coupling_block(
                    W,
                    row_slice=slice(src_start, src_start + src_size),
                    col_slice=slice(dst_start, dst_start + dst_size),
                    rng=rng,
                )

    def _fill_coupling_block(
        self,
        W: np.ndarray,
        row_slice: slice,
        col_slice: slice,
        rng: np.random.Generator,
    ) -> None:
        shape = (
            row_slice.stop - row_slice.start,
            col_slice.stop - col_slice.start,
        )
        mask = rng.random(shape) < self.coupling_density
        weights = rng.uniform(-self.coupling_strength, self.coupling_strength, shape)
        W[row_slice, col_slice] = np.where(mask, weights, 0.0)
