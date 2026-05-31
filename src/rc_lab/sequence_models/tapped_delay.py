from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rc_lab.readouts.ridge import RidgeReadout


_FEATURE_MODES = {"raw", "linear", "quadratic", "linear_quadratic"}


@dataclass
class TappedDelayPreparedSplit:
    features: np.ndarray
    targets: np.ndarray


class TappedDelayRidge:
    """
    Non-recurrent ridge baseline with explicit input taps.

    Features at time t are [u(t), u(t-1), ..., u(t-L)]. The effective washout
    for standalone full sequences is max(washout, n_lags).

    Correspondencia NVAR / NG-RC
    ----------------------------
    Con ``feature_mode="quadratic"`` y entrada escalar, la librería de
    características por muestra es:

        θ(t) = [u(t), u(t-1), …, u(t-L),  u(t)², u(t-1)², …, u(t-L)²]

    que tiene 2*(L+1) columnas. Esto constituye un **NVAR de orden 2 diagonal**:
    incluye los monomios lineales y los cuadrados individuales de cada retardo,
    pero *no* los productos cruzados u(t-i)·u(t-j) con i≠j.

    La predicción final es:
        ŷ(t) = Σᵢ aᵢ·u(t-i) + Σᵢ bᵢ·u(t-i)²

    con coeficientes entrenados por ridge (sin término constante explícito,
    ya que RidgeReadout usa fit_intercept=False).

    Esta es la forma funcional del NG-RC primitivo (Gauthier et al. 2021 §II).
    Use ``kind="ng_rc"`` en los configs del runner externo como alias semántico
    para esta configuración.

    Brecha hacia el NG-RC completo de Gauthier
    -------------------------------------------
    Para reproducir fielmente el NG-RC de Gauthier et al. (2021) faltaría:
    - Productos cruzados u(t-i)·u(t-j), i≠j (librería de monomios completa).
    - Orden polinomial configurable (k > 2).
    - Selección/poda de monomios.
    - Término constante explícito separado.
    Estas extensiones están fuera del alcance de esta versión primitiva.

    Feature modes
    -------------
    raw / linear  : θ(t) = [u(t), …, u(t-L)]                  → (L+1) columnas
    quadratic     : θ(t) = [u(t), …, u(t-L), u(t)², …, u(t-L)²] → 2*(L+1) columnas
    linear_quadratic : idéntico a quadratic (alias semántico)
    """

    def __init__(
        self,
        n_lags: int,
        ridge_param: float = 1e-6,
        feature_mode: str = "raw",
    ) -> None:
        if n_lags < 0:
            raise ValueError("n_lags must be >= 0")
        if ridge_param <= 0:
            raise ValueError("ridge_param must be > 0")
        if feature_mode not in _FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {sorted(_FEATURE_MODES)}, got {feature_mode!r}")
        self.n_lags = int(n_lags)
        self.ridge_param = float(ridge_param)
        self.feature_mode = feature_mode
        self.effective_washout: int | None = None
        self._readout: RidgeReadout | None = None
        self._n_features: int | None = None
        self._n_outputs: int | None = None

    def fit(self, u_full: np.ndarray, y_full: np.ndarray, washout: int) -> None:
        split = self.prepare_full_split(u_full, y_full, washout)
        readout = RidgeReadout(self.ridge_param)
        readout.fit(split.features, split.targets)
        self._readout = readout
        self._n_features = split.features.shape[1]
        self._n_outputs = split.targets.shape[1]
        self.effective_washout = max(washout, self.n_lags)

    def predict_full(self, u_full: np.ndarray, washout: int) -> np.ndarray:
        if self._readout is None:
            raise RuntimeError("TappedDelayRidge is not fitted")
        start = max(washout, self.n_lags)
        features = self._features_for_range(u_full, start, len(u_full))
        return self._readout.predict(features)

    def predict_prepared(self, features: np.ndarray) -> np.ndarray:
        if self._readout is None:
            raise RuntimeError("TappedDelayRidge is not fitted")
        return self._readout.predict(features)

    def prepare_full_split(
        self,
        u_full: np.ndarray,
        y_full: np.ndarray,
        washout: int,
    ) -> TappedDelayPreparedSplit:
        start = max(washout, self.n_lags)
        if len(u_full) != len(y_full):
            raise ValueError("u_full and y_full must have the same length")
        if start >= len(u_full):
            raise ValueError(
                f"Not enough samples after effective washout: len={len(u_full)}, "
                f"effective_washout={start}"
            )
        features = self._features_for_range(u_full, start, len(u_full))
        targets = np.asarray(y_full[start:], dtype=float)
        return TappedDelayPreparedSplit(features=features, targets=targets)

    def prepare_reset_scored_split(
        self,
        u_full: np.ndarray,
        y_scored: np.ndarray,
        washout: int,
    ) -> TappedDelayPreparedSplit:
        start = max(washout, self.n_lags)
        trim = start - washout
        if trim >= len(y_scored):
            raise ValueError(
                f"Not enough scored samples after effective washout: n_scored={len(y_scored)}, "
                f"washout={washout}, n_lags={self.n_lags}"
            )
        features = self._features_for_range(u_full, start, len(u_full))
        targets = np.asarray(y_scored[trim:], dtype=float)
        return TappedDelayPreparedSplit(features=features, targets=targets)

    def prepare_scored_with_history(
        self,
        history_u: np.ndarray,
        u_scored: np.ndarray,
        y_scored: np.ndarray,
    ) -> TappedDelayPreparedSplit:
        if self.n_lags == 0:
            combined = np.asarray(u_scored, dtype=float)
            start = 0
        else:
            if len(history_u) < self.n_lags:
                raise ValueError(
                    f"history_u must contain at least n_lags rows, got {len(history_u)} < {self.n_lags}"
                )
            combined = np.vstack([history_u[-self.n_lags :], u_scored])
            start = self.n_lags
        features = self._features_for_range(combined, start, len(combined))
        return TappedDelayPreparedSplit(features=features, targets=np.asarray(y_scored, dtype=float))

    def _features_for_range(self, u: np.ndarray, start: int, stop: int) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1:
            u_arr = u_arr.reshape(-1, 1)
        if start < self.n_lags:
            raise ValueError("start must be >= n_lags")
        rows = []
        for t in range(start, stop):
            taps = [u_arr[t - lag] for lag in range(0, self.n_lags + 1)]
            rows.append(np.concatenate(taps))
        features = np.asarray(rows, dtype=float)
        if self.feature_mode in ("quadratic", "linear_quadratic"):
            features = np.hstack([features, features**2])
        return features

    @property
    def n_total_params(self) -> int:
        if self._n_features is None or self._n_outputs is None:
            return 0
        return int(self._n_features * self._n_outputs)

    @property
    def n_trainable_params(self) -> int:
        return self.n_total_params

