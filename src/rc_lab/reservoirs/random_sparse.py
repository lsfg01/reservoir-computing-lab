import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs

from rc_lab.reservoirs.base import BaseReservoirBuilder, ReservoirMatrices

# Umbral de tamaño para elegir entre eigs (disperso) y eigvals (denso)
_SPARSE_THRESHOLD = 20


class RandomSparseReservoir(BaseReservoirBuilder):
    """
    Reservoir aleatorio disperso con reescalado por radio espectral.

    Parámetros del constructor
    --------------------------
    spectral_radius : parámetro de reescalado espectral (rho > 0).
                      Su relación con estabilidad y ESP es objeto de estudio
                      experimental; no se impone ningún intervalo obligatorio.
    input_scaling   : escala de Win; entradas en U(-input_scaling, input_scaling)
    sparsity        : fracción de ceros en W, en [0, 1)
    leak_rate       : alpha ∈ (0, 1]; almacenado para uso externo (ESNModel)
    bias_scaling    : escala del bias; 0.0 desactiva el bias
    """

    def __init__(
        self,
        spectral_radius: float,
        input_scaling: float,
        sparsity: float,
        leak_rate: float = 1.0,
        bias_scaling: float = 0.0,
    ) -> None:
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.bias_scaling = bias_scaling

    def build(self, N: int, n_inputs: int, seed: int) -> ReservoirMatrices:
        rng = np.random.default_rng(seed)

        # --- W dispersa ---
        density = 1.0 - self.sparsity
        # scipy.sparse.random usa RandomState; usamos seed directamente
        W = sparse_random(
            N, N,
            density=density,
            format="csr",
            random_state=int(seed),
            data_rvs=lambda s: rng.uniform(-1.0, 1.0, s),
        ).toarray()

        # Reescalado espectral
        rho_actual = self._compute_spectral_radius(W)
        if rho_actual > 0:
            W = W * (self.spectral_radius / rho_actual)

        # --- Win ---
        Win = rng.uniform(-self.input_scaling, self.input_scaling, (N, n_inputs))

        # --- bias ---
        if self.bias_scaling > 0:
            bias = rng.uniform(-self.bias_scaling, self.bias_scaling, (N,))
        else:
            bias = np.zeros(N)

        return ReservoirMatrices(W=W, Win=Win, bias=bias)

    def _compute_spectral_radius(self, W: np.ndarray) -> float:
        """
        Calcula max(|eigenvalues(W)|).
        Usa eigs(k=1) para matrices grandes, eigvals completo para pequeñas.
        """
        N = W.shape[0]
        if N > _SPARSE_THRESHOLD:
            try:
                vals = eigs(W, k=1, which="LM", return_eigenvectors=False)
                return float(np.max(np.abs(vals)))
            except Exception:
                pass  # fallback a eigvals completo
        vals = np.linalg.eigvals(W)
        return float(np.max(np.abs(vals)))
