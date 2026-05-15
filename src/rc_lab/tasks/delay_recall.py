from __future__ import annotations

import numpy as np

from rc_lab.tasks.base import BaseTask, TaskData


def delay_recall_valid_start(washout: int, kmax: int) -> int:
    """First supervised sample index for classical MC-style delay recall."""
    return int(washout) + int(kmax)


def build_delay_recall_targets(u: np.ndarray, kmax: int) -> np.ndarray:
    """Build Y[t, k-1] = u[t-k] for k=1..kmax."""
    if kmax < 1:
        raise ValueError("kmax must be >= 1")
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(-1, 1)
    if u_arr.ndim != 2 or u_arr.shape[1] != 1:
        raise ValueError(f"delay_recall expects scalar input with shape (T, 1), got {u_arr.shape}")

    y = np.zeros((u_arr.shape[0], kmax), dtype=float)
    flat = u_arr[:, 0]
    for k in range(1, kmax + 1):
        y[k:, k - 1] = flat[:-k]
    return y


class DelayRecallTask(BaseTask):
    """
    Supervised delay-recall task.

    Inputs are scalar iid samples u(t) ~ Uniform(input_low, input_high). Targets
    are multi-output delayed copies:

        y(t) = [u(t-1), u(t-2), ..., u(t-kmax)].

    To match classical Memory Capacity alignment, the first kmax samples after
    the user-provided washout are warmup-only. The effective washout stored in
    TaskData is therefore washout + kmax.
    """

    def __init__(
        self,
        kmax: int,
        input_low: float = -1.0,
        input_high: float = 1.0,
        state_policy: str = "reset",
    ) -> None:
        if kmax < 1:
            raise ValueError("kmax must be >= 1")
        if input_low >= input_high:
            raise ValueError("input_low must be < input_high")
        if state_policy not in ("reset", "carryover"):
            raise ValueError(f"state_policy invalido: {state_policy!r}")
        self._kmax = int(kmax)
        self._input_low = float(input_low)
        self._input_high = float(input_high)
        self._state_policy = state_policy

    @property
    def name(self) -> str:
        return "delay_recall"

    @property
    def primary_metric(self) -> str:
        return "memory_corr_total"

    @property
    def kmax(self) -> int:
        return self._kmax

    def generate(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        if washout < self._kmax:
            raise ValueError(
                f"DelayRecallTask requires washout >= kmax, got washout={washout}, "
                f"kmax={self._kmax}"
            )
        if self._state_policy == "reset":
            return self._generate_reset(n_train, n_val, n_test, washout, seed)
        return self._generate_carryover(n_train, n_val, n_test, washout, seed)

    def _generate_pairs(self, total: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        u = rng.uniform(self._input_low, self._input_high, size=(total, 1))
        y = build_delay_recall_targets(u, self._kmax)
        return u, y

    def _generate_reset(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        effective_washout = delay_recall_valid_start(washout, self._kmax)
        n_val_blocks = 1 if n_val > 0 else 0
        total = (
            effective_washout + n_train
            + n_val_blocks * (effective_washout + n_val)
            + effective_washout + n_test
        )
        u, y = self._generate_pairs(total, seed)

        i = 0
        u_train = u[i : i + effective_washout + n_train]
        y_train = y[i : i + effective_washout + n_train]
        i += effective_washout + n_train

        if n_val > 0:
            u_val_full = u[i : i + effective_washout + n_val]
            y_val_full = y[i : i + effective_washout + n_val]
            u_val: np.ndarray | None = u_val_full[effective_washout:]
            y_val: np.ndarray | None = y_val_full[effective_washout:]
            i += effective_washout + n_val
        else:
            u_val_full = None
            u_val, y_val = None, None

        u_test_full = u[i : i + effective_washout + n_test]
        y_test_full = y[i : i + effective_washout + n_test]

        return TaskData(
            u_train=u_train,
            y_train=y_train,
            u_val=u_val,
            y_val=y_val,
            u_test=u_test_full[effective_washout:],
            y_test=y_test_full[effective_washout:],
            washout=effective_washout,
            state_policy="reset",
            u_val_full=u_val_full,
            u_test_full=u_test_full,
        )

    def _generate_carryover(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        effective_washout = delay_recall_valid_start(washout, self._kmax)
        total = effective_washout + n_train + n_val + n_test
        u, y = self._generate_pairs(total, seed)

        i = 0
        u_train = u[i : i + effective_washout + n_train]
        y_train = y[i : i + effective_washout + n_train]
        i += effective_washout + n_train

        if n_val > 0:
            u_val: np.ndarray | None = u[i : i + n_val]
            y_val: np.ndarray | None = y[i : i + n_val]
            i += n_val
        else:
            u_val, y_val = None, None

        return TaskData(
            u_train=u_train,
            y_train=y_train,
            u_val=u_val,
            y_val=y_val,
            u_test=u[i : i + n_test],
            y_test=y[i : i + n_test],
            washout=effective_washout,
            state_policy="carryover",
            u_val_full=None,
            u_test_full=None,
        )
