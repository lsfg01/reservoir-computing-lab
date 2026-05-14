import numpy as np

from rc_lab.tasks.base import BaseTask, TaskData


_INITIAL_HISTORY_MODES = {"constant", "random_uniform"}
_STATE_POLICIES = {"reset", "carryover"}


def _validate_mackey_glass_params(
    tau: float,
    dt: float,
    beta: float,
    gamma: float,
    n: int,
    prediction_horizon: int,
    sample_stride: int,
    discard_transient: int,
    initial_history: str,
    history_low: float,
    history_high: float,
    state_policy: str | None = None,
) -> None:
    if tau <= 0:
        raise ValueError("tau debe ser > 0")
    if dt <= 0:
        raise ValueError("dt debe ser > 0")
    if beta <= 0:
        raise ValueError("beta debe ser > 0")
    if gamma <= 0:
        raise ValueError("gamma debe ser > 0")
    if n <= 0:
        raise ValueError("n debe ser > 0")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon debe ser >= 1")
    if sample_stride < 1:
        raise ValueError("sample_stride debe ser >= 1")
    if discard_transient < 0:
        raise ValueError("discard_transient debe ser >= 0")
    if initial_history not in _INITIAL_HISTORY_MODES:
        raise ValueError(
            "initial_history debe ser 'constant' o 'random_uniform', "
            f"recibido: {initial_history!r}"
        )
    if initial_history == "random_uniform" and history_low >= history_high:
        raise ValueError("history_low debe ser < history_high con random_uniform")
    if state_policy is not None and state_policy not in _STATE_POLICIES:
        raise ValueError(f"state_policy invalido: {state_policy!r}")


def _make_initial_history(
    delay_steps: int,
    seed: int,
    initial_history: str,
    history_value: float,
    history_low: float,
    history_high: float,
) -> np.ndarray:
    history_len = delay_steps + 1
    if initial_history == "constant":
        return np.full(history_len, history_value, dtype=float)

    rng = np.random.default_rng(seed)
    return rng.uniform(history_low, history_high, size=history_len)


def _integrate_mackey_glass(
    total_steps: int,
    seed: int,
    tau: float,
    dt: float,
    beta: float,
    gamma: float,
    n: int,
    initial_history: str,
    history_value: float,
    history_low: float,
    history_high: float,
) -> np.ndarray:
    """
    Integrate the Mackey-Glass delay equation with the same RK4 scheme used by
    the original task. The delayed value is kept fixed across RK4 substeps.
    """
    delay_steps = int(round(tau / dt))
    history = _make_initial_history(
        delay_steps=delay_steps,
        seed=seed,
        initial_history=initial_history,
        history_value=history_value,
        history_low=history_low,
        history_high=history_high,
    )
    history_len = delay_steps + 1
    x = np.empty(history_len + total_steps, dtype=float)
    x[:history_len] = history

    def mg_deriv(x_now: float, x_delayed: float) -> float:
        return beta * x_delayed / (1.0 + x_delayed**n) - gamma * x_now

    for i in range(history_len, history_len + total_steps):
        x_now = x[i - 1]
        x_del = x[i - 1 - delay_steps]

        k1 = mg_deriv(x_now, x_del)
        k2 = mg_deriv(x_now + 0.5 * dt * k1, x_del)
        k3 = mg_deriv(x_now + 0.5 * dt * k2, x_del)
        k4 = mg_deriv(x_now + dt * k3, x_del)

        x[i] = x_now + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x[history_len : history_len + total_steps]


def _generate_subsampled_series(
    n_pairs: int,
    seed: int,
    tau: float,
    dt: float,
    beta: float,
    gamma: float,
    n: int,
    prediction_horizon: int,
    sample_stride: int,
    discard_transient: int,
    initial_history: str,
    history_value: float,
    history_low: float,
    history_high: float,
) -> np.ndarray:
    """
    Return enough subsampled points to build exactly n_pairs scored pairs.

    prediction_horizon is expressed on the subsampled series. For example,
    sample_stride=10 and prediction_horizon=84 predicts 840 internal dt steps
    ahead when dt=0.1.
    """
    required_subsampled_points = n_pairs + prediction_horizon
    total_steps = (
        discard_transient
        + sample_stride * (required_subsampled_points - 1)
        + 1
    )
    raw = _integrate_mackey_glass(
        total_steps=total_steps,
        seed=seed,
        tau=tau,
        dt=dt,
        beta=beta,
        gamma=gamma,
        n=n,
        initial_history=initial_history,
        history_value=history_value,
        history_low=history_low,
        history_high=history_high,
    )
    series = raw[discard_transient::sample_stride]
    return series[:required_subsampled_points]


def _build_prediction_pairs(
    series: np.ndarray,
    prediction_horizon: int,
    n_pairs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_pairs is None:
        n_pairs = len(series) - prediction_horizon
    if n_pairs < 0 or len(series) < n_pairs + prediction_horizon:
        raise ValueError("series no contiene puntos suficientes para el horizonte")

    u = series[:n_pairs].reshape(-1, 1)
    y = series[prediction_horizon : prediction_horizon + n_pairs].reshape(-1, 1)
    return u, y


def generate_mackey_glass(
    T: int,
    seed: int,
    tau: float = 17,
    dt: float = 0.1,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    prediction_horizon: int = 1,
    sample_stride: int = 1,
    discard_transient: int = 0,
    initial_history: str = "constant",
    history_value: float = 0.9,
    history_low: float = 1.1,
    history_high: float = 1.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Mackey-Glass prediction pairs.

    Defaults preserve the original task: constant history 0.9, no extra
    transient discard, no subsampling, and one-step prediction on the integrated
    dt=0.1 series. With sample_stride > 1, prediction_horizon is applied after
    subsampling.
    """
    _validate_mackey_glass_params(
        tau=tau,
        dt=dt,
        beta=beta,
        gamma=gamma,
        n=n,
        prediction_horizon=prediction_horizon,
        sample_stride=sample_stride,
        discard_transient=discard_transient,
        initial_history=initial_history,
        history_low=history_low,
        history_high=history_high,
    )
    series = _generate_subsampled_series(
        n_pairs=T,
        seed=seed,
        tau=tau,
        dt=dt,
        beta=beta,
        gamma=gamma,
        n=n,
        prediction_horizon=prediction_horizon,
        sample_stride=sample_stride,
        discard_transient=discard_transient,
        initial_history=initial_history,
        history_value=history_value,
        history_low=history_low,
        history_high=history_high,
    )
    return _build_prediction_pairs(series, prediction_horizon, n_pairs=T)


class MackeyGlassTask(BaseTask):
    """
    Mackey-Glass prediction task.

    state_policy="reset":
        [washout][train][washout][val][washout][test]

    state_policy="carryover":
        [washout][train][val][test]

    The protocol and TaskData shapes are unchanged from the original task; only
    the generated Mackey-Glass series and (u, y) horizon can be configured.
    """

    def __init__(
        self,
        tau: float = 17,
        dt: float = 0.1,
        beta: float = 0.2,
        gamma: float = 0.1,
        n: int = 10,
        prediction_horizon: int = 1,
        sample_stride: int = 1,
        discard_transient: int = 0,
        initial_history: str = "constant",
        history_value: float = 0.9,
        history_low: float = 1.1,
        history_high: float = 1.3,
        state_policy: str = "reset",
    ) -> None:
        _validate_mackey_glass_params(
            tau=tau,
            dt=dt,
            beta=beta,
            gamma=gamma,
            n=n,
            prediction_horizon=prediction_horizon,
            sample_stride=sample_stride,
            discard_transient=discard_transient,
            initial_history=initial_history,
            history_low=history_low,
            history_high=history_high,
            state_policy=state_policy,
        )
        self._tau = tau
        self._dt = dt
        self._beta = beta
        self._gamma = gamma
        self._n = n
        self._prediction_horizon = prediction_horizon
        self._sample_stride = sample_stride
        self._discard_transient = discard_transient
        self._initial_history = initial_history
        self._history_value = history_value
        self._history_low = history_low
        self._history_high = history_high
        self._state_policy = state_policy

    @property
    def name(self) -> str:
        return "mackey_glass"

    @property
    def primary_metric(self) -> str:
        return "nmse"

    @property
    def prediction_horizon(self) -> int:
        return self._prediction_horizon

    @property
    def sample_stride(self) -> int:
        return self._sample_stride

    @property
    def discard_transient(self) -> int:
        return self._discard_transient

    @property
    def initial_history(self) -> str:
        return self._initial_history

    def generate(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        if self._state_policy == "reset":
            return self._generate_reset(n_train, n_val, n_test, washout, seed)
        return self._generate_carryover(n_train, n_val, n_test, washout, seed)

    def _generate_pairs(self, total: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return generate_mackey_glass(
            total,
            seed=seed,
            tau=self._tau,
            dt=self._dt,
            beta=self._beta,
            gamma=self._gamma,
            n=self._n,
            prediction_horizon=self._prediction_horizon,
            sample_stride=self._sample_stride,
            discard_transient=self._discard_transient,
            initial_history=self._initial_history,
            history_value=self._history_value,
            history_low=self._history_low,
            history_high=self._history_high,
        )

    def _generate_reset(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        """Reset protocol: each split has an explicit washout block."""
        n_val_blocks = 1 if n_val > 0 else 0
        total = (
            washout + n_train
            + n_val_blocks * (washout + n_val)
            + washout + n_test
        )
        u, y = self._generate_pairs(total, seed=seed)

        i = 0
        u_train = u[i : i + washout + n_train]
        y_train = y[i : i + washout + n_train]
        i += washout + n_train

        if n_val > 0:
            u_val_full = u[i : i + washout + n_val]
            u_val: np.ndarray | None = u_val_full[washout:]
            y_val: np.ndarray | None = y[i + washout : i + washout + n_val]
            i += washout + n_val
        else:
            u_val_full = None
            u_val, y_val = None, None

        u_test_full = u[i : i + washout + n_test]
        u_test = u_test_full[washout:]
        y_test = y[i + washout : i + washout + n_test]

        return TaskData(
            u_train=u_train,
            y_train=y_train,
            u_val=u_val,
            y_val=y_val,
            u_test=u_test,
            y_test=y_test,
            washout=washout,
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
        """Carryover protocol: one continuous scored trajectory."""
        total = washout + n_train + n_val + n_test
        u, y = self._generate_pairs(total, seed=seed)

        i = 0
        u_train = u[i : i + washout + n_train]
        y_train = y[i : i + washout + n_train]
        i += washout + n_train

        if n_val > 0:
            u_val: np.ndarray | None = u[i : i + n_val]
            y_val: np.ndarray | None = y[i : i + n_val]
            i += n_val
        else:
            u_val, y_val = None, None

        u_test = u[i : i + n_test]
        y_test = y[i : i + n_test]

        return TaskData(
            u_train=u_train,
            y_train=y_train,
            u_val=u_val,
            y_val=y_val,
            u_test=u_test,
            y_test=y_test,
            washout=washout,
            state_policy="carryover",
            u_val_full=None,
            u_test_full=None,
        )
