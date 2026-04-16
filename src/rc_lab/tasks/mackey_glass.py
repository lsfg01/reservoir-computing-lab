import numpy as np

from rc_lab.tasks.base import BaseTask, TaskData


def generate_mackey_glass(
    T: int,
    seed: int,
    tau: int = 17,
    dt: float = 0.1,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera la serie de Mackey-Glass mediante integración numérica (RK4).

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    La tarea de predicción es: dado x(t), predecir x(t+1).

    Nota sobre determinismo: la serie es completamente determinista para un
    conjunto fijo de parámetros (tau, dt, beta, gamma, n) y condición inicial
    constante 0.9. El parámetro `seed` se acepta por consistencia de interfaz
    con el resto de tareas, pero no altera el dataset generado.

    El calentamiento implícito de `history_len = int(tau/dt) + 1` pasos con
    condición inicial constante es suficiente para que la serie entre en régimen.

    Devuelve (u, y) con shapes (T, 1), donde u = x(t) e y = x(t+1).
    """
    history_len = int(tau / dt) + 1
    total_steps = T + 1  # +1 para construir y = x(t+1)

    x = np.full(history_len + total_steps, 0.9)

    def mg_deriv(x_now: float, x_delayed: float) -> float:
        return beta * x_delayed / (1.0 + x_delayed ** n) - gamma * x_now

    for i in range(history_len, history_len + total_steps):
        x_now = x[i - 1]
        x_del = x[i - 1 - history_len + 1]

        k1 = mg_deriv(x_now, x_del)
        k2 = mg_deriv(x_now + 0.5 * dt * k1, x_del)
        k3 = mg_deriv(x_now + 0.5 * dt * k2, x_del)
        k4 = mg_deriv(x_now + dt * k3, x_del)

        x[i] = x_now + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    series = x[history_len : history_len + total_steps]
    u = series[:-1].reshape(-1, 1)
    y = series[1:].reshape(-1, 1)
    return u, y


class MackeyGlassTask(BaseTask):
    """
    Tarea Mackey-Glass: predecir x(t+1) dado x(t).

    Protocolo de splits
    -------------------
    state_policy="reset" (por defecto):
        La serie se organiza como:
            [washout][n_train][washout][n_val][washout][n_test]
        Cada split tiene su propio bloque de warmup de longitud `washout`.
        u_val_full y u_test_full transportan los bloques completos con warmup.
        u_val, y_val, u_test, y_test son los arrays puntuados (post-warmup).

    state_policy="carryover":
        La serie se organiza como:
            [washout][n_train][n_val][n_test]
        El estado del reservoir se propaga entre splits contiguos.
        u_val_full y u_test_full son None.
    """

    def __init__(
        self,
        tau: int = 17,
        dt: float = 0.1,
        state_policy: str = "reset",
    ) -> None:
        if state_policy not in ("reset", "carryover"):
            raise ValueError(f"state_policy inválido: {state_policy!r}")
        self._tau = tau
        self._dt = dt
        self._state_policy = state_policy

    @property
    def name(self) -> str:
        return "mackey_glass"

    @property
    def primary_metric(self) -> str:
        return "nmse"

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
        else:
            return self._generate_carryover(n_train, n_val, n_test, washout, seed)

    def _generate_reset(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        washout: int,
        seed: int,
    ) -> TaskData:
        """
        Protocolo reset: cada split tiene warmup propio de longitud `washout`.

        Serie total generada:
            [washout][n_train][washout][n_val][washout][n_test]
        """
        n_val_blocks = 1 if n_val > 0 else 0
        total = (
            washout + n_train
            + n_val_blocks * (washout + n_val)
            + washout + n_test
        )
        u, y = generate_mackey_glass(total, seed=seed, tau=self._tau, dt=self._dt)

        i = 0
        u_train = u[i : i + washout + n_train]
        y_train = y[i : i + washout + n_train]
        i += washout + n_train

        if n_val > 0:
            u_val_full = u[i : i + washout + n_val]
            y_val_full = y[i : i + washout + n_val]
            u_val: np.ndarray | None = u_val_full[washout:]
            y_val: np.ndarray | None = y_val_full[washout:]
            i += washout + n_val
        else:
            u_val_full = None
            u_val, y_val = None, None

        u_test_full = u[i : i + washout + n_test]
        y_test_full = y[i : i + washout + n_test]
        u_test = u_test_full[washout:]
        y_test = y_test_full[washout:]

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
        """
        Protocolo carryover: serie continua, estado propagado entre splits.

        Serie total generada:
            [washout][n_train][n_val][n_test]
        """
        total = washout + n_train + n_val + n_test
        u, y = generate_mackey_glass(total, seed=seed, tau=self._tau, dt=self._dt)

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
